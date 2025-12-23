from model import Generator
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import piq
import lpips
#import dpo_config
#import dpo_trainer

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
model1 = "exp_pie_b4"
model2 = "base"
generator = Generator().to(device)
generator.load_state_dict(torch.load(f"./results/{model1}/g-best_lpips.pth"))
generator.eval()

generator2 = Generator().to(device)
generator2.load_state_dict(torch.load(f"./results/{model2}/g-last.pth"))
generator2.eval()

# LPIPS 모델 초기화
lpips_model = lpips.LPIPS(net='alex').to(device)

# LPIPS 계산 함수
def calculate_lpips(pred, target):
    pred_np = pred.detach()
    target_np = target.detach()
    return lpips_model(pred_np, target_np).mean().item()

x = Image.open(opt.image_path)
transform = transforms.Compose([transforms.Resize((x.size[1]//4, x.size[0]//4), Image.BICUBIC),transforms.ToTensor()]) #이미지에 맞게 resize 지금 1/4
transform1 = transforms.Compose([transforms.Resize((x.size[1]//4, x.size[0]//4), Image.BICUBIC),transforms.ToTensor()])
transform2 = transforms.Compose([transforms.Resize(((x.size[1]//4)*4, (x.size[0]//4)*4), Image.BICUBIC),transforms.ToTensor()])
# Prepare input
image_tensor = Variable(transform(x)).to(device).unsqueeze(0) 
fn = opt.image_path.split("/")[-1]
# Save image
#save_image(transform(x), f"results/test/pieonly/{fn}-input.png")

# Upsample image
with torch.no_grad():
    sr_image = generator(image_tensor)

# Upsample image
with torch.no_grad():
    sr_image2 = generator2(image_tensor)

imgs_hr = transform2(x).unsqueeze(0).to(device)
clamp_sr = torch.clamp(sr_image, 0, 1)
pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=x.size[0])(imgs_hr, clamp_sr)
pie=torch.Tensor.cpu(pieapp_loss)
pie=pie.cuda()
pie=abs(torch.mean(pie))
print("pieapp:", pie)
lpips_value = calculate_lpips(imgs_hr, clamp_sr)
print("lpips: ", lpips_value)
psnr_value = piq.psnr(imgs_hr, clamp_sr)
print("psnr: ",psnr_value.item())
ssim_value = piq.ssim(imgs_hr, clamp_sr)
print("ssim: ", ssim_value.item())

# Save image
save_image(sr_image, f"results/test/pieonly/sr-{model1}-{fn}")

clamp_sr2 = torch.clamp(sr_image2, 0, 1)
pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=x.size[0])(imgs_hr, clamp_sr2)
pie=torch.Tensor.cpu(pieapp_loss)
pie=pie.cuda()
pie=abs(torch.mean(pie))
print("base pieapp:", pie)
lpips_value = calculate_lpips(imgs_hr, clamp_sr2)
print("base lpips: ", lpips_value)
psnr_value = piq.psnr(imgs_hr, clamp_sr2)
print("base psnr: ",psnr_value.item())
ssim_value = piq.ssim(imgs_hr, clamp_sr2)
print("base ssim: ", ssim_value.item())

# Save image
save_image(sr_image2, f"results/test/pieonly/sr-{model2}-{fn}")