## Isomorphic MRI Reconsturction
### Installation
```bash
git clone https://github.com/Ever2after/mri-recon.git
cd mri-recon
```
```bash
conda create -n mri python=3.10
conda activate mri
pip install -r requirements.txt
```

### Data
```
mri-recon/
│
├── data/
│   ├── train/
│   │   ├── *.nii.gz
│   │   └── ...
│   └── test/
│       ├── *.nii.gz
│       └── ...
│
├── models/
│   └── autoencoder.py 
│
├── train.py                   
│
├── requirements.txt
└── README.md
```

### Training
```python
python train.py
```