import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_skimage

from model import Generator
from dataset import BaseDataset
from config import upscale_factor

# ─── 파서 정의 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="DIV2K_valid_HR",
                    help="테스트할 데이터셋 폴더 이름")
parser.add_argument("--batch_size",  type=int,   default=1,
                    help="배치 크기")
parser.add_argument("--n_cpu",       type=int,   default=8,
                    help="DataLoader에 사용할 CPU 스레드 수")
opt = parser.parse_args()

# ─── 디바이스 설정 ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Generator 불러오기 ──────────────────────────────────────────────────────
generator = Generator().to(device)
generator.load_state_dict(torch.load(
    "./results/ref_only_dpo_max-min_nobackward/g-last.pth",
    map_location=device
))
generator.eval()

# ─── DataLoader 준비 ─────────────────────────────────────────────────────────
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

# ─── SSIM 저장 리스트 ────────────────────────────────────────────────────────
ssim_values = []

# ─── 평가 루프 ───────────────────────────────────────────────────────────────
for i, (lr, hr) in enumerate(dataloader):
    lr = lr.to(device)
    hr = hr.to(device)

    # SR 생성
    with torch.no_grad():
        sr = generator(lr)
    sr = torch.clamp(sr, 0.0, 1.0)

    # tensor → numpy 변환 ([0,1] → [0,255], uint8)
    hr_np = (hr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    sr_np = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # win_size 계산 (최대 7, 홀수, 최소 3)
    h, w = hr_np.shape[:2]
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    # SSIM 계산 (scikit-image 방식)
    ssim_val, _ = ssim_skimage(
        hr_np,
        sr_np,
        data_range=255,
        channel_axis=2,   # RGB 채널이 마지막 축일 때
        win_size=win_size,
        full=True
    )

    ssim_values.append(ssim_val)
    print(f"[Batch {i:3d}] SSIM: {ssim_val:.4f}")

# ─── 전체 평균 SSIM 출력 ─────────────────────────────────────────────────────
mean_ssim = np.mean(ssim_values)
print(f"\n=== 전체 평균 SSIM: {mean_ssim:.4f} ===")
