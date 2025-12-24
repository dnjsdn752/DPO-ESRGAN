
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

writing = True

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
#parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)

pth_mode = ['g-last.pth', 'g-best_lpips.pth']

# LPIPS calculation function
def calculate_lpips(target, pred):
    pred_np = pred.detach()
    target_np = target.detach()
    return lpips_model(pred_np, target_np).mean().item()

path = "./results/noref_lpips_min-max"

for pth in pth_mode:
    # Initialize generator and discriminator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(os.path.join(path, pth)))
    generator.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    image_size = 128
    mode = "original_size_valid"

    dataset_name = "Set14"
    test_dataset = BaseDataset("../data/%s" % dataset_name, image_size, upscale_factor, mode=mode)
    dataloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    content_list = []
    psnr_value_list = []
    ssim_value_list = []
    lpips_value_list = []
    pieapp_list = []

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
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=clamp_sr.shape[2])(hr.detach(), clamp_sr) 
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))
        psnr_value = piq.psnr(hr, clamp_sr)
        ssim_value = piq.ssim(hr, clamp_sr)
        lpips_value = calculate_lpips(hr, clamp_sr)
        squeezed_tensor = clamp_sr.squeeze(0)

        content_list.append(loss_content.item())
        psnr_value_list.append(psnr_value.item())
        ssim_value_list.append(ssim_value.item())
        lpips_value_list.append(lpips_value)
        pieapp_list.append(pie.item())

        print(
            "[batch: %d , content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % (
                i,
                loss_content.item(),
                psnr_value.item(),
                ssim_value.item(),
                lpips_value,
                pie.item()
            )
        )

    print(
            "%s: mean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % ( 
                dataset_name,
                np.mean(content_list),
                np.mean(psnr_value_list),
                np.mean(ssim_value_list),
                np.mean(lpips_value_list),
                np.mean(pieapp_list)
            )
        )

    if writing:
        results_file_path = os.path.join(path, 'test.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write("%s\nmean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f] \n"
                    % (
                        dataset_name,
                        np.mean(content_list),
                        np.mean(psnr_value_list),
                        np.mean(ssim_value_list),
                        np.mean(lpips_value_list),
                        np.mean(pieapp_list)
                    )
                )
    dataset_name = "Set5"
    test_dataset = BaseDataset("../data/%s" % dataset_name, image_size, upscale_factor, mode=mode)
    dataloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    content_list = []
    psnr_value_list = []
    ssim_value_list = []
    lpips_value_list = []
    pieapp_list = []

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
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=clamp_sr.shape[2])(hr.detach(), clamp_sr) 
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))
        psnr_value = piq.psnr(hr, clamp_sr)
        ssim_value = piq.ssim(hr, clamp_sr)
        lpips_value = calculate_lpips(hr, clamp_sr)
        squeezed_tensor = clamp_sr.squeeze(0)

        content_list.append(loss_content.item())
        psnr_value_list.append(psnr_value.item())
        ssim_value_list.append(ssim_value.item())
        lpips_value_list.append(lpips_value)
        pieapp_list.append(pie.item())

        print(
            "[batch: %d , content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % (
                i,
                loss_content.item(),
                psnr_value.item(),
                ssim_value.item(),
                lpips_value,
                pie.item()
            )
        )

    print(
            "%s: mean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % ( 
                dataset_name,
                np.mean(content_list),
                np.mean(psnr_value_list),
                np.mean(ssim_value_list),
                np.mean(lpips_value_list),
                np.mean(pieapp_list)
            )
        )

    if writing:
        results_file_path = os.path.join(path, 'test.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write("%s\nmean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f] \n"
                    % (
                        dataset_name,
                        np.mean(content_list),
                        np.mean(psnr_value_list),
                        np.mean(ssim_value_list),
                        np.mean(lpips_value_list),
                        np.mean(pieapp_list)
                    )
                )
    dataset_name = "DIV2K_valid_HR"
    test_dataset = BaseDataset("../data/%s" % dataset_name, image_size, upscale_factor, mode=mode)
    dataloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    content_list = []
    psnr_value_list = []
    ssim_value_list = []
    lpips_value_list = []
    pieapp_list = []

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
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=clamp_sr.shape[2])(hr.detach(), clamp_sr) 
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))
        psnr_value = piq.psnr(hr, clamp_sr)
        ssim_value = piq.ssim(hr, clamp_sr)
        lpips_value = calculate_lpips(hr, clamp_sr)
        squeezed_tensor = clamp_sr.squeeze(0)

        content_list.append(loss_content.item())
        psnr_value_list.append(psnr_value.item())
        ssim_value_list.append(ssim_value.item())
        lpips_value_list.append(lpips_value)
        pieapp_list.append(pie.item())

        print(
            "[batch: %d , content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % (
                i,
                loss_content.item(),
                psnr_value.item(),
                ssim_value.item(),
                lpips_value,
                pie.item()
            )
        )

    print(
            "%s: mean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % ( 
                dataset_name,
                np.mean(content_list),
                np.mean(psnr_value_list),
                np.mean(ssim_value_list),
                np.mean(lpips_value_list),
                np.mean(pieapp_list)
            )
        )

    if writing:
        results_file_path = os.path.join(path, 'test.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write("%s\nmean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f] \n"
                    % (
                        dataset_name,
                        np.mean(content_list),
                        np.mean(psnr_value_list),
                        np.mean(ssim_value_list),
                        np.mean(lpips_value_list),
                        np.mean(pieapp_list)
                    )
                )
    dataset_name = "BSD100"
    test_dataset = BaseDataset("../data/%s" % dataset_name, image_size, upscale_factor, mode=mode)
    dataloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    content_list = []
    psnr_value_list = []
    ssim_value_list = []
    lpips_value_list = []
    pieapp_list = []

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
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=clamp_sr.shape[2])(hr.detach(), clamp_sr) 
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))
        psnr_value = piq.psnr(hr, clamp_sr)
        ssim_value = piq.ssim(hr, clamp_sr)
        lpips_value = calculate_lpips(hr, clamp_sr)
        squeezed_tensor = clamp_sr.squeeze(0)

        content_list.append(loss_content.item())
        psnr_value_list.append(psnr_value.item())
        ssim_value_list.append(ssim_value.item())
        lpips_value_list.append(lpips_value)
        pieapp_list.append(pie.item())

        print(
            "[batch: %d , content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % (
                i,
                loss_content.item(),
                psnr_value.item(),
                ssim_value.item(),
                lpips_value,
                pie.item()
            )
        )

    print(
            "%s: mean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % ( 
                dataset_name,
                np.mean(content_list),
                np.mean(psnr_value_list),
                np.mean(ssim_value_list),
                np.mean(lpips_value_list),
                np.mean(pieapp_list)
            )
        )

    if writing:
        results_file_path = os.path.join(path, 'test.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write("%s\nmean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f] \n"
                    % (
                        dataset_name,
                        np.mean(content_list),
                        np.mean(psnr_value_list),
                        np.mean(ssim_value_list),
                        np.mean(lpips_value_list),
                        np.mean(pieapp_list)
                    )
                )
    dataset_name = "URban100"
    test_dataset = BaseDataset("../data/%s" % dataset_name, image_size, upscale_factor, mode=mode)
    dataloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    content_list = []
    psnr_value_list = []
    ssim_value_list = []
    lpips_value_list = []
    pieapp_list = []

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
        pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=clamp_sr.shape[2])(hr.detach(), clamp_sr) 
        pie=torch.Tensor.cpu(pieapp_loss)
        pie=pie.cuda()
        pie=abs(torch.mean(pie))
        psnr_value = piq.psnr(hr, clamp_sr)
        ssim_value = piq.ssim(hr, clamp_sr)
        lpips_value = calculate_lpips(hr, clamp_sr)
        squeezed_tensor = clamp_sr.squeeze(0)

        content_list.append(loss_content.item())
        psnr_value_list.append(psnr_value.item())
        ssim_value_list.append(ssim_value.item())
        lpips_value_list.append(lpips_value)
        pieapp_list.append(pie.item())

        print(
            "[batch: %d , content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % (
                i,
                loss_content.item(),
                psnr_value.item(),
                ssim_value.item(),
                lpips_value,
                pie.item()
            )
        )

    print(
            "%s: mean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f]"
            % ( 
                dataset_name,
                np.mean(content_list),
                np.mean(psnr_value_list),
                np.mean(ssim_value_list),
                np.mean(lpips_value_list),
                np.mean(pieapp_list)
            )
        )

    if writing:
        results_file_path = os.path.join(path, 'test.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write("%s\nmean: [content: %f, psnr: %f, ssim: %f, lpips: %f, pie: %f] \n"
                    % (
                        dataset_name,
                        np.mean(content_list),
                        np.mean(psnr_value_list),
                        np.mean(ssim_value_list),
                        np.mean(lpips_value_list),
                        np.mean(pieapp_list)
                    )
                )