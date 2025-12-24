import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_skimage

from model import Generator
from dataset import BaseDataset
from config import upscale_factor

# ─── Parser definition ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="DIV2K_valid_HR",
                    help="Dataset folder name to test")
parser.add_argument("--batch_size",  type=int,   default=1,
                    help="Batch size")
parser.add_argument("--n_cpu",       type=int,   default=8,
                    help="Number of CPU threads for DataLoader")
opt = parser.parse_args()

# ─── Device setting ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load Generator ──────────────────────────────────────────────────────
generator = Generator().to(device)
generator.load_state_dict(torch.load(
    "./results/ref_only_dpo_max-min_nobackward/g-last.pth",
    map_location=device
))
generator.eval()

# ─── Prepare DataLoader ─────────────────────────────────────────────────────────
image_size   = 128
dataset_path = os.path.join("../data", opt.dataset_name)
test_dataset = BaseDataset(
    dataset_path,
    image_size,
    upscale_factor,
    "original_size_valid"
)
dataloader = DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# ─── SSIM save list ────────────────────────────────────────────────────────
ssim_values = []

# ─── Evaluation loop ───────────────────────────────────────────────────────────────
for i, (lr, hr) in enumerate(dataloader):
    lr = lr.to(device)
    hr = hr.to(device)

    # Generate SR
    with torch.no_grad():
        sr = generator(lr)
    sr = torch.clamp(sr, 0.0, 1.0)

    # Convert tensor -> numpy ([0,1] -> [0,255], uint8)
    hr_np = (hr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    sr_np = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Calculate win_size (max 7, odd, min 3)
    h, w = hr_np.shape[:2]
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    # Calculate SSIM (scikit-image method)
    ssim_val, _ = ssim_skimage(
        hr_np,
        sr_np,
        data_range=255,
        channel_axis=2,   # When RGB channel is the last axis
        win_size=win_size,
        full=True
    )

    ssim_values.append(ssim_val)
    print(f"[Batch {i:3d}] SSIM: {ssim_val:.4f}")

# ─── Print overall average SSIM ─────────────────────────────────────────────────────
mean_ssim = np.mean(ssim_values)
print(f"\n=== Overall Average SSIM: {mean_ssim:.4f} ===")
