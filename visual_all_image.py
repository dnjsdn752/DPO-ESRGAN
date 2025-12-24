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

# Folder path where the image is saved
datasetname = 'URban100'
folder_path = f"../data/{datasetname}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
model1 = "ref_dpo_max-min" #last
model2 = "ref_dpo_min-max_nobackward" #best
os.makedirs(f"results/test/pieonly/{model1}/{datasetname}", exist_ok=True)
os.makedirs(f"results/test/pieonly/{model2}/{datasetname}", exist_ok=True)

generator = Generator().to(device)
generator.load_state_dict(torch.load(f"./results/{model1}/g-last.pth"))
generator.eval()

generator2 = Generator().to(device)
generator2.load_state_dict(torch.load(f"./results/{model2}/g-last.pth"))
generator2.eval()

# LPIPS model initialization
lpips_model = lpips.LPIPS(net='alex').to(device)

# LPIPS calculation function
def calculate_lpips(pred, target):
    pred_np = pred.detach()
    target_np = target.detach()
    return lpips_model(pred_np, target_np).mean().item()

for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)  
        x = Image.open(image_path)

        transform = transforms.Compose([transforms.Resize((x.size[1]//4, x.size[0]//4), Image.BICUBIC),transforms.ToTensor()]) 
        transform1 = transforms.Compose([transforms.Resize((x.size[1]//4, x.size[0]//4), Image.BICUBIC),transforms.ToTensor()])
        transform2 = transforms.Compose([transforms.Resize(((x.size[1]//4)*4, (x.size[0]//4)*4), Image.BICUBIC),transforms.ToTensor()])
        # Prepare input
        image_tensor = Variable(transform(x)).to(device).unsqueeze(0) 
        fn = image_path.split("/")[-1]
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
        print(fn)
        print("pieapp:", pie)
        lpips_value = calculate_lpips(imgs_hr, clamp_sr)
        print("lpips: ", lpips_value)
        psnr_value = piq.psnr(imgs_hr, clamp_sr)
        print("psnr: ",psnr_value.item())
        ssim_value = piq.ssim(imgs_hr, clamp_sr)
        print("ssim: ", ssim_value.item())
        
        results_file_path = os.path.join(f"results/test/pieonly/{model1}", f'{model1}_results.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write(fn)
            f.write(f"\npieapp: {pie:.6f} lpips: {lpips_value:.6f} PSNR: {psnr_value:.6f} ssim: {ssim_value:.6f}.\n\n")
            

        # Save image
        #save_image(sr_image, f"results/test/pieonly/{model1}/{datasetname}/sr-{model1}-{fn}")
        
        clamp_sr2 = torch.clamp(sr_image2, 0, 1)
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=x.size[0])(imgs_hr, clamp_sr2)
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))
        print(fn)
        print("base pieapp:", pie)
        lpips_value = calculate_lpips(imgs_hr, clamp_sr2)
        print("base lpips: ", lpips_value)
        psnr_value = piq.psnr(imgs_hr, clamp_sr2)
        print("base psnr: ",psnr_value.item())
        ssim_value = piq.ssim(imgs_hr, clamp_sr2)
        print("base ssim: ", ssim_value.item())
        results_file_path = os.path.join(f"results/test/pieonly/{model2}", f'{model2}_results.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write(fn)
            f.write(f"\npieapp: {pie:.6f} lpips: {lpips_value:.6f} PSNR: {psnr_value:.6f} ssim: {ssim_value:.6f}.\n\n")

        # Save image
        save_image(sr_image2, f"results/test/pieonly/{model2}/{datasetname}/sr-{model2}-{fn}")