import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib


def percentile_norm_01(x: np.ndarray, p_low=1.0, p_high=99.0, eps=1e-6) -> np.ndarray:
    """Clip to [p1,p99] then map to [0,1]. Works per-slice."""
    lo, hi = np.percentile(x, [p_low, p_high])
    x = (x - lo) / (hi - lo + eps)
    return np.clip(x, 0.0, 1.0)


def center_crop_or_pad_2d(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """img: (H,W) -> (outH,outW) using center crop/pad with zeros."""
    H, W = img.shape
    outH, outW = out_hw

    # crop
    top = max((H - outH) // 2, 0)
    left = max((W - outW) // 2, 0)
    cropped = img[top:top + min(outH, H), left:left + min(outW, W)]

    # pad
    padH = outH - cropped.shape[0]
    padW = outW - cropped.shape[1]
    pad_top = max(padH // 2, 0)
    pad_bottom = max(padH - pad_top, 0)
    pad_left = max(padW // 2, 0)
    pad_right = max(padW - pad_left, 0)

    return np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0.0)

class Rot90AndIntensity:
    """
    Apply same transform to a triplet to keep slice alignment.
    - Random k*90 deg rotation
    - Random intensity scale and shift (simple, robust)
    """
    def __init__(
        self,
        p_rot: float = 1.0,
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
        intensity_shift_range: Tuple[float, float] = (-0.05, 0.05),
        clamp: Tuple[float, float] = (0.0, 1.0),
    ):
        self.p_rot = p_rot
        self.smin, self.smax = intensity_scale_range
        self.bmin, self.bmax = intensity_shift_range
        self.cmin, self.cmax = clamp

    def __call__(self, x_prev: torch.Tensor, x_mid: torch.Tensor, x_next: torch.Tensor):
        # x_*: (1,H,W) or (C,H,W)
        if torch.rand(()) < self.p_rot:
            k = int(torch.randint(low=0, high=4, size=()).item())
            if k != 0:
                x_prev = torch.rot90(x_prev, k, dims=(-2, -1))
                x_mid  = torch.rot90(x_mid,  k, dims=(-2, -1))
                x_next = torch.rot90(x_next, k, dims=(-2, -1))

        s = torch.empty((), dtype=x_mid.dtype).uniform_(self.smin, self.smax)
        b = torch.empty((), dtype=x_mid.dtype).uniform_(self.bmin, self.bmax)
        x_prev = x_prev * s + b
        x_mid  = x_mid  * s + b
        x_next = x_next * s + b

        x_prev = x_prev.clamp(self.cmin, self.cmax)
        x_mid  = x_mid.clamp(self.cmin, self.cmax)
        x_next = x_next.clamp(self.cmin, self.cmax)
        return x_prev, x_mid, x_next


class NiiTripletDataset(Dataset):
    """
    Builds triplets (k-1,k,k+1) along axis=2 (z).
    Each __getitem__ returns (x_prev, x_mid, x_next) as torch.float32 in [0,1], shape (1,H,W).
    """

    def __init__(
        self,
        nii_paths: List[Path],
        axis: int = 2,
        out_hw: Optional[Tuple[int, int]] = None,  # e.g. (160,192) or (192,192)
        transform=None,  # triplet-level transform (same for prev/mid/next)
        p_low: float = 1.0,
        p_high: float = 99.0,
        cache_policy: str = "none",  # "none" | "header" (keeps shapes only)
    ):
        self.nii_paths = list(nii_paths)
        self.axis = axis
        self.out_hw = out_hw
        self.transform = transform
        self.p_low = p_low
        self.p_high = p_high
        self.cache_policy = cache_policy

        # Index map: global_idx -> (file_idx, k)
        # where k is the mid-slice index along `axis`, valid range [1, D-2]
        self.index: List[Tuple[int, int]] = []

        self._shapes: Dict[int, Tuple[int, ...]] = {}
        for fi, p in enumerate(self.nii_paths):
            img = nib.load(str(p))
            shape = tuple(img.shape)
            if cache_policy == "header":
                self._shapes[fi] = shape

            D = shape[self.axis]
            # valid mids: 1..D-2
            for k in range(1, D - 1):
                self.index.append((fi, k))

    def __len__(self):
        return len(self.index)

    def _load_volume(self, path: Path) -> np.ndarray:
        # float32 saves RAM
        vol = nib.load(str(path)).get_fdata(dtype=np.float32)
        # ensure 3D
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {vol.ndim}D: {path}")
        return vol

    def _get_slice(self, vol: np.ndarray, k: int) -> np.ndarray:
        # axis=2 => vol[:,:,k]
        if self.axis == 0:
            sl = vol[k, :, :]
        elif self.axis == 1:
            sl = vol[:, k, :]
        elif self.axis == 2:
            sl = vol[:, :, k]
        else:
            raise ValueError("axis must be 0/1/2")
        return sl

    def __getitem__(self, idx: int):
        fi, k = self.index[idx]
        path = self.nii_paths[fi]

        vol = self._load_volume(path)

        s_prev = self._get_slice(vol, k - 1)
        s_mid  = self._get_slice(vol, k)
        s_next = self._get_slice(vol, k + 1)

        # per-slice robust normalization -> [0,1]
        s_prev = percentile_norm_01(s_prev, self.p_low, self.p_high)
        s_mid  = percentile_norm_01(s_mid,  self.p_low, self.p_high)
        s_next = percentile_norm_01(s_next, self.p_low, self.p_high)

        # optional shape fix
        if self.out_hw is not None:
            
            # print("Before crop/pad:"
            #       f" s_prev: {s_prev.shape},"
            #       f" s_mid: {s_mid.shape},"
            #       f" s_next: {s_next.shape}")  # debug line
            
            s_prev = center_crop_or_pad_2d(s_prev, self.out_hw)
            s_mid  = center_crop_or_pad_2d(s_mid,  self.out_hw)
            s_next = center_crop_or_pad_2d(s_next, self.out_hw)
            
            # print("After crop/pad:"
            #       f" s_prev: {s_prev.shape},"
            #       f" s_mid: {s_mid.shape},"
            #       f" s_next: {s_next.shape}")  # debug line

        # to torch: (1,H,W)
        x_prev = torch.from_numpy(s_prev).unsqueeze(0).to(torch.float32)
        x_mid  = torch.from_numpy(s_mid ).unsqueeze(0).to(torch.float32)
        x_next = torch.from_numpy(s_next).unsqueeze(0).to(torch.float32)

        if self.transform is not None:
            x_prev, x_mid, x_next = self.transform(x_prev, x_mid, x_next)
            
        # print("after transform: ", x_prev.shape, x_mid.shape, x_next.shape)  # debug line

        return x_prev, x_mid, x_next


def list_nii(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))


def make_loaders(
    datapath: str,
    batch_size: int = 16,
    num_workers: int = 4,
    out_hw: Optional[Tuple[int, int]] = (160, 192),  # your native size; set None to keep original
    transform=None,
):
    root = Path(datapath)
    train_paths = list_nii(root / "train")
    test_paths  = list_nii(root / "test")

    if len(train_paths) == 0:
        raise FileNotFoundError(f"No NIfTI files found in {root/'train'}")
    if len(test_paths) == 0:
        print(f"[Warn] No NIfTI files found in {root/'test'} (using train only).")

    train_ds = NiiTripletDataset(train_paths, axis=2, out_hw=out_hw, transform=transform)
    test_ds  = NiiTripletDataset(test_paths,  axis=2, out_hw=out_hw, transform=None) if len(test_paths) else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    ) if test_ds is not None else None

    return train_loader, test_loader




import matplotlib.pyplot as plt

def debug_one_batch(loader, save_path="debug_triplet.png"):
    x_prev, x_mid, x_next = next(iter(loader))
    print("x_prev:", x_prev.shape, x_prev.min().item(), x_prev.max().item())
    print("x_mid :", x_mid.shape,  x_mid.min().item(),  x_mid.max().item())
    print("x_next:", x_next.shape, x_next.min().item(), x_next.max().item())

    # 첫 샘플만 그리기
    a = x_prev[0,0].numpy()
    b = x_mid[0,0].numpy()
    c = x_next[0,0].numpy()

    plt.figure(figsize=(9,3))
    for i, (img, title) in enumerate([(a,"prev"), (b,"mid"), (c,"next")]):
        plt.subplot(1,3,i+1)
        plt.imshow(img.T, cmap="gray", origin="lower")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    train_loader, test_loader = make_loaders(
        datapath="../../../ARPA/restore_165504/ADNI_Nifti_QC_N4",
        batch_size=4,
        num_workers=2,
        out_hw=(160,192),
        transform=None,
    )

    print("Train loader:")
    debug_one_batch(train_loader, save_path="debug_triplet_train.png")

    if test_loader is not None:
        print("Test loader:")
        debug_one_batch(test_loader, save_path="debug_triplet_test.png")