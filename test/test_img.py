import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import piq
import lpips
from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LPIPS 모델 초기화
lpips_model = lpips.LPIPS(net='alex').to(device)
# LPIPS 계산 함수
def calculate_lpips(target, pred):
    pred_np = pred.detach()
    target_np = target.detach()
    return lpips_model(pred_np, target_np).mean().item()

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return img

def to_tensor(img):
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # (1, C, H, W)

def evaluate_images_with_piq(original_dir, sr_dir, device='cpu'):
    psnr_scores, ssim_scores, lpips_scores, Pie_scores = [], [], [], []

    image_names = sorted(os.listdir(original_dir))  # 원본 이미지 이름 기준

    for orig_name in image_names:
        orig_path = os.path.join(original_dir, orig_name)
        
        sr_path = os.path.join(sr_dir, orig_name)
        
        # 이미지 이름이 out.png로 끝나는경우 밑에 3줄 주석해제, 위에 줄은 주석
        base_name = os.path.splitext(orig_name)[0]
        sr_name = f"{base_name}_esrgan_DP.png"
        sr_path = os.path.join(sr_dir, sr_name)

        sr_img = load_image(sr_path)
        orig_img = load_image(orig_path, size=sr_img.size)  # ✅ SR 이미지 크기에 맞춰 리사이즈
        
        # 텐서 변환
        sr_tensor = to_tensor(sr_img).to(device)
        orig_tensor = to_tensor(orig_img).to(device)

        # PSNR, SSIM, LPIPS 계산
        psnr = piq.psnr(sr_tensor, orig_tensor, data_range=1.0).item()
        orig_np = np.array(orig_img)           # uint8, shape (H, W, 3)
        sr_np   = np.array(sr_img)
        # --- win_size 자동 계산: min(height, width) 이하의 최대 홀수
        h, w = orig_np.shape[:2]
        max_ws = min(7, h, w)         # 기본 SSIM 윈도우 크기 7
        if max_ws % 2 == 0:
            max_ws -= 1               # 짝수이면 하나 빼서 홀수로
        win_size = max(3, max_ws)     # 최소 3이상

        # SSIM 계산 (skimage ≥0.19 버전)
        ssim, _ = ssim_skimage(
            orig_np,
            sr_np,
            data_range=255,
            # multichannel=True,              # < skimage 0.19
            channel_axis=2,                   # ≥ skimage 0.19
            win_size=win_size,
            full=True
        )
        
        lpips_val = calculate_lpips(sr_tensor , orig_tensor)  # LPIPS는 [-1, 1] 범위 필요
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=sr_tensor.shape[2])(orig_tensor.detach(), sr_tensor)
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        lpips_scores.append(lpips_val)
        Pie_scores.append(pie)

        print(f"{orig_name} | PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips_val:.4f}, Pie: {pie:.4f}")

    # 평균 출력
    print("\n=== 평균 결과 ===")
    print(f"평균 PSNR: {sum(psnr_scores)/len(psnr_scores):.4f}")
    print(f"평균 SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"평균 LPIPS: {sum(lpips_scores)/len(lpips_scores):.4f}")
    print(f"평균 Pie: {sum(Pie_scores)/len(Pie_scores):.4f}")

# 사용 예시
original_dir = "../data/Set5"
sr_dir = "results/test/pieonly/esrgan_dp/Set5"
evaluate_images_with_piq(original_dir, sr_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
