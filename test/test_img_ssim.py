import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return img

def evaluate_ssim_only(original_dir, sr_dir):
    ssim_scores = []
    image_names = sorted(os.listdir(original_dir))

    for name in image_names:
        # 파일 경로
        orig_path = os.path.join(original_dir, name)
        sr_path   = os.path.join(sr_dir, name)  # sr 파일명이 같다고 가정

        # 이미지 이름이 out.png로 끝나는경우 밑에 3줄 주석해제, 위에 줄은 주석
        base_name = os.path.splitext(name)[0]
        sr_name = f"sr-ref_lpips_max-min-{base_name}.png"
        sr_path = os.path.join(sr_dir, sr_name)

        
        # 이미지 로드 (SR은 원본 크기에 맞추기)
        orig_img = load_image(orig_path)
        sr_img   = load_image(sr_path, size=orig_img.size)

        # NumPy 배열로 변환
        orig_np = np.array(orig_img)  # (H, W, 3), uint8
        sr_np   = np.array(sr_img)

        # win_size 계산 (최대 7, 이미지 크기에 맞춰 홀수)
        h, w = orig_np.shape[:2]
        win_size = min(7, h, w)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(win_size, 3)

        # SSIM 계산
        score, _ = ssim_skimage(
            orig_np,
            sr_np,
            data_range=255,
            channel_axis=2,
            win_size=win_size,
            full=True
        )
        ssim_scores.append(score)
        print(f"{name} | SSIM: {score:.4f}")

    # 평균 SSIM 출력
    avg = sum(ssim_scores) / len(ssim_scores)
    print(f"\n=== 평균 SSIM: {avg:.4f} ===")

# 사용 예시
original_dir = "../data/URban100"
sr_dir       = "results/test/pieonly/ref_lpips_max-min/URban100"
evaluate_ssim_only(original_dir, sr_dir)
