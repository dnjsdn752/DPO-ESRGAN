import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import piq
import lpips
from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)
# LPIPS calculation function
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

    image_names = sorted(os.listdir(original_dir))  # Based on original image names

    for orig_name in image_names:
        orig_path = os.path.join(original_dir, orig_name)
        
        sr_path = os.path.join(sr_dir, orig_name)
        
        # Uncomment below 3 lines if image name ends with out.png, comment out the line above
        base_name = os.path.splitext(orig_name)[0]
        sr_name = f"{base_name}_esrgan_DP.png"
        sr_path = os.path.join(sr_dir, sr_name)

        sr_img = load_image(sr_path)
        orig_img = load_image(orig_path, size=sr_img.size)  # Resize to match SR image size
        
        # Convert to tensor
        sr_tensor = to_tensor(sr_img).to(device)
        orig_tensor = to_tensor(orig_img).to(device)

        # Calculate PSNR, SSIM, LPIPS
        psnr = piq.psnr(sr_tensor, orig_tensor, data_range=1.0).item()
        orig_np = np.array(orig_img)           # uint8, shape (H, W, 3)
        sr_np   = np.array(sr_img)
        # --- Auto calculate win_size: max odd number <= min(height, width)
        h, w = orig_np.shape[:2]
        max_ws = min(7, h, w)         # 기본 SSIM 윈도우 크기 7
        if max_ws % 2 == 0:
            max_ws -= 1               # If even, subtract 1 to make it odd
        win_size = max(3, max_ws)     # Minimum 3 or more

        # Calculate SSIM (skimage >= 0.19 version)
        ssim, _ = ssim_skimage(
            orig_np,
            sr_np,
            data_range=255,
            # multichannel=True,              # < skimage 0.19
            channel_axis=2,                   # ≥ skimage 0.19
            win_size=win_size,
            full=True
        )
        
        lpips_val = calculate_lpips(sr_tensor , orig_tensor)  # LPIPS needs [-1, 1] range
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=sr_tensor.shape[2])(orig_tensor.detach(), sr_tensor)
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        lpips_scores.append(lpips_val)
        Pie_scores.append(pie)

        print(f"{orig_name} | PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips_val:.4f}, Pie: {pie:.4f}")

    # Print average
    print("\n=== Average Result ===")
    print(f"Mean PSNR: {sum(psnr_scores)/len(psnr_scores):.4f}")
    print(f"Mean SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"Mean LPIPS: {sum(lpips_scores)/len(lpips_scores):.4f}")
    print(f"Mean Pie: {sum(Pie_scores)/len(Pie_scores):.4f}")

# Example usage
original_dir = "../data/Set5"
sr_dir = "results/test/pieonly/esrgan_dp/Set5"
evaluate_images_with_piq(original_dir, sr_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
