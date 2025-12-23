
import argparse
import os
import numpy as np
import math
import itertools
import sys
import piq
import lpips
import basicsr

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataset import *
from config import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Set14", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
#parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize generator and discriminator
generator = Generator().to(device)
generator.load_state_dict(torch.load("./results/base/g-last.pth"))
generator.eval()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
image_size = 128

test_dataset = BaseDataset("../data/%s" % opt.dataset_name, image_size, upscale_factor, "valid")
dataloader = DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# LPIPS 모델 초기화
lpips_model = lpips.LPIPS(net='alex').to(device)

# LPIPS 계산 함수
def calculate_lpips(target, pred):
    pred_np = pred.detach()
    target_np = target.detach()
    return lpips_model(pred_np, target_np).mean().item()

content_list = []
psnr_value_list = []
ssim_value_list = []
lpips_value_list = []
pieapp_list = []
niqe_list = []

for i, (lr, hr) in enumerate(dataloader):
    
    # Copy the data to the specified device.
    lr = lr.to(device)
    hr = hr.to(device)

    # Generate a high resolution image from low resolution input
    with torch.no_grad():
        sr = generator(lr)

    loss_content = content_criterion(sr, hr.detach())

    #indicator
    clamp_sr = torch.clamp(sr, 0, 1)
    pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=clamp_sr.shape[2])(hr.detach(), clamp_sr) #원본이미지와 비교하기 위해서 model통과한 결과도 denormalize ->pie결과가 모두동일...
    pie=torch.Tensor.cpu(pieapp_loss)
    pie=pie.cuda()
    pie=abs(torch.mean(pie))
    psnr_value = piq.psnr(hr, clamp_sr)
    ssim_value = piq.ssim(hr, clamp_sr)
    lpips_value = calculate_lpips(hr, clamp_sr)
    squeezed_tensor = clamp_sr.squeeze(0)
    ndarray = squeezed_tensor.cpu().numpy()
    niqe_value = basicsr.metrics.niqe.calculate_niqe(ndarray, crop_border=4, input_order="CHW")

    content_list.append(loss_content.item())
    psnr_value_list.append(psnr_value.item())
    ssim_value_list.append(ssim_value.item())
    lpips_value_list.append(lpips_value)
    pieapp_list.append(pie.item())
    niqe_list.append(niqe_value)

    print(
        "[batch: %d , content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f, NIQE: %f]"
        % (
            i,
            loss_content.item(),
            psnr_value.item(),
            ssim_value.item(),
            lpips_value,
            pie.item(),
            niqe_value
        )
    )

print(
        "mean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f, NIQE: %f]"
        % (
            np.mean(content_list),
            np.mean(psnr_value_list),
            np.mean(ssim_value_list),
            np.mean(lpips_value_list),
            np.mean(pieapp_list),
            np.mean(niqe_list)
        )
    )