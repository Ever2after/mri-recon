import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Import your stuff here
# -------------------------
# 1) dataloader
# from your_dataloader_file import make_loaders, Rot90AndIntensity
# 2) model
# from your_model_file import ASIAutoencoder

# If you already have latent mix helper in model file, you can import it too.
@torch.no_grad()
def latent_convex_mix(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    a = float(alpha)
    return (1.0 - a) * z0 + a * z1


# -------------------------
# LPIPS (preferred) with safe fallback
# -------------------------
class LPIPSLoss(nn.Module):
    """
    Expects inputs in [0,1]. For grayscale, repeats to 3 channels.
    Uses lpips package if available; otherwise VGG-feature MSE fallback.
    """
    def __init__(self, net: str = "vgg"):
        super().__init__()
        try:
            import lpips  # pip install lpips
            self.fn = lpips.LPIPS(net=net)
            self._has_lpips = True
        except Exception:
            self.fn = None
            self._has_lpips = False

        if not self._has_lpips:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
            for p in vgg.parameters():
                p.requires_grad_(False)
            self.vgg = vgg
            self.layers = [3, 8, 15, 22, 29]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x3 = x.repeat(1, 3, 1, 1)
            y3 = y.repeat(1, 3, 1, 1)
        else:
            x3, y3 = x, y

        if self._has_lpips:
            x_in = x3 * 2.0 - 1.0
            y_in = y3 * 2.0 - 1.0
            return self.fn(x_in, y_in).mean()

        # fallback VGG feature MSE
        h1, h2 = x3, y3
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            h1 = layer(h1)
            h2 = layer(h2)
            if i in self.layers:
                loss = loss + F.mse_loss(h1, h2)
        return loss


# -------------------------
# Training config
# -------------------------
@dataclass
class TrainConfig:
    datapath: str
    device: str = "cuda"
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4

    # paper-like
    lr: float = 1e-5
    lambda_synth: float = 0.05
    alpha: float = 0.5

    amp: bool = True
    grad_clip_norm: Optional[float] = None

    out_hw: Optional[Tuple[int, int]] = (160, 192)  # keep native, change if you want
    save_dir: str = "./checkpoints"
    save_every: int = 1
    log_every: int = 50

    log_dir: str = "./logs"
    debug_dir: str = "./debug"
    debug_every: int = 200  # steps
    resume: Optional[str] = None  # path to ckpt


# -------------------------
# Debug image saving
# -------------------------
@torch.no_grad()
def save_debug_triplet(
    x_mid: torch.Tensor,
    x_hat_mid: torch.Tensor,
    x_hat_mix: torch.Tensor,
    out_path: str,
    max_items: int = 4,
):
    """
    Saves a grid: rows = items, cols = [GT, recon, synth]
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x_mid = x_mid.detach().cpu()
    x_hat_mid = x_hat_mid.detach().cpu()
    x_hat_mix = x_hat_mix.detach().cpu()

    B = min(x_mid.shape[0], max_items)

    plt.figure(figsize=(9, 3 * B))
    for i in range(B):
        gt = x_mid[i, 0].numpy()
        rc = x_hat_mid[i, 0].numpy()
        sy = x_hat_mix[i, 0].numpy()

        for j, (img, title) in enumerate([(gt, "GT"), (rc, "Recon"), (sy, "Synth")]):
            ax = plt.subplot(B, 3, i * 3 + j + 1)
            ax.imshow(img.T, cmap="gray", origin="lower")
            ax.set_title(f"{title} [{i}]")
            ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------
# Core train/val loops
# -------------------------
def train_one_epoch(model, loader, test_loader, optimizer, mse_loss, lpips_loss, cfg, scaler, epoch: int):
    model.train()
    device = cfg.device

    run_recon = 0.0
    run_synth = 0.0
    run_total = 0.0

    for step, (x_prev, x_mid, x_next) in enumerate(loader):
        
        print(f"Epoch {epoch:03d} Step {step+1:05d}/{len(loader):05d}")  # debug line
        
        global_step = epoch * len(loader) + step
        
        x_prev = x_prev.to(device, non_blocking=True)
        x_mid  = x_mid.to(device,  non_blocking=True)
        x_next = x_next.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = (scaler is not None) and cfg.amp and (device.startswith("cuda"))
        with torch.cuda.amp.autocast(enabled=use_amp):
            # recon
            x_hat_mid, _ = model(x_mid)
            loss_recon = mse_loss(x_hat_mid, x_mid)

            # synth: mix neighbors in latent
            z_prev = model.encode(x_prev)
            z_next = model.encode(x_next)
            z_mix = latent_convex_mix(z_prev, z_next, cfg.alpha)
            x_hat_mix = model.decode(z_mix)

            loss_synth = lpips_loss(x_mid, x_hat_mix)
            loss = loss_recon + cfg.lambda_synth * loss_synth

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        run_recon += float(loss_recon.detach().cpu())
        run_synth += float(loss_synth.detach().cpu())
        run_total += float(loss.detach().cpu())

        # log
        if (step + 1) % cfg.log_every == 0:
            denom = cfg.log_every
            print(
                f"[Epoch {epoch:03d} | Step {step+1:05d}/{len(loader):05d}] "
                f"[Train] recon={run_recon/denom:.6f} synth={run_synth/denom:.6f} total={run_total/denom:.6f}"
            )
            # log file
            train_log_path = os.path.join(cfg.log_dir, "train_log.txt")
            with open(train_log_path, "a") as f:
                # train 
                f.write(
                    f"[Epoch {epoch:03d} | Step {step+1:05d}/{len(loader):05d}]\n"
                    f"{loss_recon.item():.6f},{loss_synth.item():.6f},{loss.item():.6f},\n"
                )
            
            run_recon = run_synth = run_total = 0.0
            
            # validation
            # if test_loader is not None:
            #     metrics = validate(model, test_loader, mse_loss, lpips_loss, cfg)
            #     print(f"  [Val] recon={metrics['recon']:.6f} synth={metrics['synth']:.6f} total={metrics['total']:.6f}")
            
            #     val_log_path = os.path.join(cfg.log_dir, "val_log.txt")
            #     with open(val_log_path, "a") as f:
            #         # val
            #         f.write(
            #             f"{global_step},{epoch},{step+1},"
            #             f"{metrics['recon']:.6f},{metrics['synth']:.6f},{metrics['total']:.6f}\n"
            #     )

        # debug image
        if cfg.debug_every > 0 and (global_step % cfg.debug_every == 0):
            out_path = os.path.join(cfg.debug_dir, f"ep{epoch:03d}_step{step:05d}.png")
            save_debug_triplet(x_mid, x_hat_mid, x_hat_mix, out_path)


@torch.no_grad()
def validate(model, loader, mse_loss, lpips_loss, cfg) -> Dict[str, float]:
    model.eval()
    device = cfg.device

    sum_recon, sum_synth, sum_total = 0.0, 0.0, 0.0
    n = 0

    for x_prev, x_mid, x_next in loader:
        x_prev = x_prev.to(device, non_blocking=True)
        x_mid  = x_mid.to(device,  non_blocking=True)
        x_next = x_next.to(device, non_blocking=True)

        x_hat_mid, _ = model(x_mid)
        loss_recon = mse_loss(x_hat_mid, x_mid)

        z_prev = model.encode(x_prev)
        z_next = model.encode(x_next)
        z_mix = latent_convex_mix(z_prev, z_next, cfg.alpha)
        x_hat_mix = model.decode(z_mix)

        loss_synth = lpips_loss(x_mid, x_hat_mix)
        loss = loss_recon + cfg.lambda_synth * loss_synth

        bs = x_mid.shape[0]
        sum_recon += float(loss_recon.cpu()) * bs
        sum_synth += float(loss_synth.cpu()) * bs
        sum_total += float(loss.cpu()) * bs
        n += bs

    return {"recon": sum_recon / max(n, 1), "synth": sum_synth / max(n, 1), "total": sum_total / max(n, 1)}


def save_ckpt(path: str, model, optimizer, scaler, epoch: int, best_val: float, cfg: TrainConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val": best_val,
            "cfg": cfg.__dict__,
        },
        path,
    )


def load_ckpt(path: str, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    return start_epoch, best_val


def main(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.debug_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device
    print(f"[Device] {device}")

    # -------------------------
    # Build loaders
    # -------------------------

    # IMPORTANT: set these imports correctly
    from data.utils import make_loaders, Rot90AndIntensity
    from models.autoencoder import ASIAutoencoder

    transform = Rot90AndIntensity()
    train_loader, test_loader = make_loaders(
        cfg.datapath,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        out_hw=cfg.out_hw,
        transform=None
    )

    # -------------------------
    # Build model & losses
    # -------------------------
    model = ASIAutoencoder(in_ch=1, out_ch=1, negative_slope=0.2, apply_paper_init=True).to(device)

    mse_loss = nn.MSELoss()
    lpips_loss = LPIPSLoss(net="vgg").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))

    start_epoch = 0
    best_val = float("inf")

    if cfg.resume is not None and os.path.exists(cfg.resume):
        start_epoch, best_val = load_ckpt(cfg.resume, model, optimizer, scaler)
        print(f"[Resume] {cfg.resume} | start_epoch={start_epoch} best_val={best_val:.6f}")

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(start_epoch, cfg.epochs):
        train_one_epoch(model, train_loader, test_loader, optimizer, mse_loss, lpips_loss, cfg, scaler, epoch)

        if test_loader is not None:
            metrics = validate(model, test_loader, mse_loss, lpips_loss, cfg)
            print(f"[Val {epoch:03d}] recon={metrics['recon']:.6f} synth={metrics['synth']:.6f} total={metrics['total']:.6f}")
            # log file
            val_log_path = os.path.join(cfg.log_dir, "val_log.txt")
            with open(val_log_path, "a") as f:
                # val
                f.write(
                    f"{(epoch+1)*len(train_loader)},{epoch},-1,"
                    f"{metrics['recon']:.6f},{metrics['synth']:.6f},{metrics['total']:.6f}\n"
                )

            is_best = metrics["total"] < best_val
            if is_best:
                best_val = metrics["total"]
                save_ckpt(os.path.join(cfg.save_dir, "best.pt"), model, optimizer, scaler, epoch, best_val, cfg)

        if (epoch + 1) % cfg.save_every == 0:
            save_ckpt(os.path.join(cfg.save_dir, "last.pt"), model, optimizer, scaler, epoch, best_val, cfg)
            
    # loss curve plot
    


if __name__ == "__main__":
    cfg = TrainConfig(
        datapath="adni",   # <-- CHANGE THIS (contains train/ and test/)
        device="cuda",
        epochs=10,
        batch_size=16,
        num_workers=4,
        lr=1e-5,
        lambda_synth=0.05,
        alpha=0.5,
        out_hw=(160, 192),
        save_dir="./checkpoints",
        debug_dir="./debug",
        debug_every=200,
        log_dir="./logs",
        log_every=50,
        resume=None,
    )
    main(cfg)
