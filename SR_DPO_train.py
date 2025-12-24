# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# Original code available at: https://github.com/cyun-404/PieESRGAN?tab=readme-ov-file
#
# Modified by Wonwoo Yun in 2025 for research purposes.
# This modified version is licensed under the MIT License.
# ==============================================================================

# ============================================================================
# File description: Realize the model training function.
# ============================================================================
from torch.utils.data import DataLoader

from config import *
from dataset import BaseDataset
from torch.utils.tensorboard import SummaryWriter
log_dir = "./log_dir"
writer = SummaryWriter(log_dir)
import piq
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import lpips
#from utils.image_utils import *

beta: float = 0.1
label_smoothing: float = 0.0
#epsilon = 1e-8  # Very small value
os.makedirs(exp_dir1, exist_ok=True)
os.makedirs(exp_dir2, exist_ok=True)

#ref generator
ref_generator = Generator().to(device)
ref_generator.load_state_dict(torch.load("./results/pie/p-best.pth"))
for param in ref_generator.parameters():
    param.requires_grad = False
ref_generator.eval()

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)
# Set model parameters not to update
for param in lpips_model.parameters():
    param.requires_grad = False

# Initialize PieAPP model
pie_model = piq.PieAPP(reduction='none', stride=128, enable_grad=True)

# LPIPS calculation function
def calculate_lpips(pred, target):
    pred_np = pred.detach()
    target_np = target.detach()
    return lpips_model(pred_np, target_np).mean().item()

def pass_lpips(target, pred):
   
    lpips_output = lpips_model(pred, target)
    lpips_value = torch.mean(lpips_output)
    return lpips_value

def train_generator(train_dataloader, epoch) -> None:
    
    # Calculate how many iterations there are under epoch.
    batches = len(train_dataloader)
    # Set generator network in training mode.
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # Copy the data to the specified device.
        lr = lr.to(device)
        hr = hr.to(device)
        # Initialize the gradient of the generator model.
        generator.zero_grad()
        # Generate super-resolution images.
        sr = generator(lr)
        # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        pixel_loss = pixel_criterion(sr, hr)
        # Update the weights of the generator model.
        pixel_loss.backward()
        p_optimizer.step()
        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Generator/Loss", pixel_loss.item(), iters)
        # Print the loss function every ten iterations and the last iteration in this epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Train Epoch[{epoch + 1:04d}/{p_epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {pixel_loss.item():.6f}.")


def train_adversarial(train_dataloader, epoch) -> None:
   
    # Calculate how many iterations there are under Epoch.
    batches = len(train_dataloader)
    # Set adversarial network in training mode.
    #discriminator.train()
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # Copy the data to the specified device.
        lr = lr.to(device)
        hr = hr.to(device)
        
        label_size = lr.size(0)
        # Create label. Set the real sample label to 1, and the false sample label to 0.
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)
  
        # Initialize the gradient of the discriminator model.
        discriminator.zero_grad()
        # Generate super-resolution images.
        sr = generator(lr)
        # Calculate the loss of the discriminator model on the high-resolution image.
        hr_output = discriminator(hr)
        sr_output = discriminator(sr.detach())
        d_loss_hr = adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
        d_loss_hr.backward()
        d_hr = hr_output.mean().item()
        # Calculate the loss of the discriminator model on the super-resolution image.
        hr_output = discriminator(hr)
        sr_output = discriminator(sr.detach())
        d_loss_sr = adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
        d_loss_sr.backward()
        d_sr1 = sr_output.mean().item()
        # Update the weights of the discriminator model.
        d_loss = d_loss_hr + d_loss_sr
        d_optimizer.step()
        
        # Initialize the gradient of the generator model.
        generator.zero_grad()
        # Generate super-resolution images.
        sr = generator(lr)
        ref_sr = ref_generator(lr)

        # Calculate the loss of the discriminator model on the super-resolution image.
        hr_output = discriminator(hr.detach())
        sr_output = discriminator(sr)
        # Perceptual loss=0.01 * pixel loss + 1.0 * content loss + 0.005 * adversarial loss.
        pixel_loss = pixel_weight * pixel_criterion(sr, hr.detach())
        content_loss = content_weight * content_criterion(sr, hr.detach())
        adversarial_loss = adversarial_weight * adversarial_criterion(sr_output - torch.mean(hr_output), real_label)
        # Update the weights of the generator model.
        clamp_sr = torch.clamp(sr, 0, 1)
        clamp_ref_sr = torch.clamp(ref_sr, 0, 1)
        
        
        # Pass batch through value model
        value_outputs = []

        for i in range(clamp_sr.size(0)):
            # Input for current batch
            current_input = clamp_sr[i].unsqueeze(0)  # Reshape to (1, 3, 128, 128)
            hr_input = hr[i].unsqueeze(0)
            ref_input = clamp_ref_sr[i].unsqueeze(0)
            
            pieapp_loss = pie_model(hr_input.detach(), current_input)
            ref_pie_loss = pie_model(hr_input.detach(), ref_input)
            pie=abs(torch.mean(pieapp_loss))
            ref_pie = abs(torch.mean(ref_pie_loss))
            pie_inverse = -pie 
            ref_pie_inverse = -ref_pie
            value_outputs.append((pie_inverse,ref_pie_inverse))
            
            # When using LPIPS 
            # lpips_value = pass_lpips(hr_input.detach(), current_input)  
            # ref_lpips_value = pass_lpips(hr_input.detach(), ref_input) 
            # lpips_inverse = -lpips_value
            # ref_lpips_inverse = -ref_lpips_value
            # value_outputs.append((lpips_inverse, ref_lpips_inverse))  # Add output value to list
            

        # Total generator loss

        max_value = max(value_outputs, key=lambda x: x[0])
        min_value = min(value_outputs, key=lambda x: x[0])
        #dpo loss
        logits = torch.log((max_value[0].detach())/max_value[1]) - torch.log(min_value[0]/min_value[1])
        loss_dpo = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
        )

        g_loss = content_loss + adversarial_loss +  pixel_loss + loss_dpo
        g_loss.backward()#.backward()
        g_optimizer.step()
        
        d_sr2 = sr_output.mean().item()

        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        #writer.add_scalar("Train_Adversarial/D_Loss", d_loss.item(), iters)
        #writer.add_scalar("Train_Adversarial/G_Loss", g_loss.item(), iters)
        #writer.add_scalar("Train_Adversarial/pie_Loss", pie.mean().item(),iters)
        #writer.add_scalar("Train_Adversarial/D_HR", d_hr, iters)
        #writer.add_scalar("Train_Adversarial/D_SR1", d_sr1, iters)
        #writer.add_scalar("Train_Adversarial/D_SR2", d_sr2, iters)
        #writer.add_scalar("Train_Adversarial/piexel_loss", pixel_loss.item(), iters)
        #writer.add_scalar("Train_Adversarial/content_loss", content_loss.item(), iters)
        #writer.add_scalar("Train_Adversarial/adversarial_loss", adversarial_loss.item(), iters)
        #writer.add_scalar("Train_Adversarial/dpo_loss", loss_dpo.item(), iters)

        # Print the loss function every ten iterations and the last iteration in this epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(
                  #f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  ##
                  f"Train Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.6f}/{d_sr2:.6f}.\n"
                  f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.mean().item():.6f}\n"
                  f"pixel Loss: {pixel_loss.item():.6f} "
                  f"pie: {pie.mean().item():.6f}.\n"
                  f"content_loss Loss: {content_loss.item():.6f} adversarial_loss Loss: {adversarial_loss.item():.6f} \n"
                    "[Dpo loss: %f] [chosen_rewards: %f, rejected_rewards: %f]"
                    % (
                        loss_dpo.item(),
                        max_value[0].item(),
                        min_value[0].item(),
                    )
                )  
        if (index +1) ==batches:
            results_file_path = os.path.join(exp_dir2, 'loss.txt')
            with open(results_file_path, 'a') as f:  # Append mode
                f.write(f"Train Epoch[{epoch + 1:04d}/{epochs:04d}]"
                    f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.mean().item():.6f} "
                    f"pixel Loss: {pixel_loss.item():.6f} "
                    f"pie: {pie.mean().item():.6f} "
                    f"content_loss Loss: {content_loss.item():.6f} adversarial_loss Loss: {adversarial_loss.item():.6f} "
                        "[Dpo loss: %f] [chosen_rewards: %f, rejected_rewards: %f] \n"
                        % (
                            loss_dpo.item(),
                            max_value[0].item(),
                            min_value[0].item(),
                        ))

def validate(valid_dataloader, epoch, stage) -> float:

    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    generator.eval()
    # Initialize the evaluation index.
    total_psnr_value = 0.0
    total_pie_value = 0.0
    total_lpips_value = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            # Copy the data to the specified device.
            lr = lr.to(device)
            hr = hr.to(device)
            # Generate super-resolution images.
            sr = generator(lr)
            # Calculate the PSNR indicator.
            mse_loss = psnr_criterion(sr, hr)
            psnr_value = 10 * torch.log10(1 / mse_loss).item()

            clamp_sr = torch.clamp(sr, 0, 1)
            pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=128)(hr.detach(), clamp_sr) 
            pie=torch.Tensor.cpu(pieapp_loss)
            pie=pie.cuda()
            pie_value=abs(torch.mean(pie))
            lpips_value = calculate_lpips(hr, clamp_sr)

            total_psnr_value += psnr_value
            total_pie_value += pie_value.item()
            total_lpips_value += lpips_value

        avg_psnr_value = total_psnr_value / batches
        avg_pie_value = total_pie_value / batches
        avg_lpips_value = total_lpips_value / batches
        # Write the value of each round of verification indicators into Tensorboard.
        if stage == "generator":
            writer.add_scalar("Val_Generator/PSNR", avg_psnr_value, epoch + 1)
        elif stage == "adversarial":
            writer.add_scalar("Val_Adversarial/PSNR", avg_psnr_value, epoch + 1)
            writer.add_scalar("Val_Adversarial/pie", avg_pie_value, epoch + 1)
            writer.add_scalar("Val_Adversarial/lpips", avg_lpips_value, epoch + 1)
        
        results_file_path = os.path.join(exp_dir2, 'results.txt')
        with open(results_file_path, 'a') as f:  # Append mode
            f.write(f"Valid stage: {stage} Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.6f} avg pie: {avg_pie_value:.6f} avg lpips: {avg_lpips_value:.6f}.\n")

        # Print evaluation indicators.
        print(f"Valid stage: {stage} Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.6f} avg pie: {avg_pie_value:.6f} avg lpips: {avg_lpips_value:.6f}.\n")

    return avg_psnr_value, avg_pie_value, avg_lpips_value


def main() -> None:
    
    # Load the dataset.
    train_dataset = BaseDataset(train_dir, image_size, upscale_factor, "train")
    valid_dataset = BaseDataset(valid_dir, image_size, upscale_factor, "valid")
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    # Check whether the training progress of the last abnormal end is restored, for example, the power is
    # cut off in the middle of the training.
    if resume:
        print("Resuming...")
        if resume_p_weight != "":
            generator.load_state_dict(torch.load(resume_p_weight))
        else:
            discriminator.load_state_dict(torch.load(resume_d_weight))
            generator.load_state_dict(torch.load(resume_g_weight))

    # Initialize the evaluation indicators for the training stage of the generator model.
    best_psnr_value = 0.0
    # Train the generative network stage.
    for epoch in range(start_p_epoch, p_epochs):
        # Train each epoch for generator network.
        train_generator(train_dataloader, epoch)
        # Verify each epoch for generator network.
        psnr_value = validate(valid_dataloader, epoch, "generator")
        # Determine whether the performance of the generator network under epoch is the best.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the weight of the generator network under epoch. If the performance of the generator network under epoch
        # is best, save a file ending with `-best.pth` in the `results` directory.
        if (epoch+1)%50==0:
            torch.save(generator.state_dict(), os.path.join(exp_dir1, f"p_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-best.pth"))
        # Adjust the learning rate of the generator model.
        p_scheduler.step()

    # Save the weight of the last generator network under epoch in this stage.
    #torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-last.pth"))
    

    # Initialize the evaluation index of the adversarial network training phase.
    best_psnr_value = 0.0
    best_lpips_value = 100.0
    best_pie_value = 100.0
    # Load the model weights with the best indicators in the previous round of training.
    generator.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))
    # Training the adversarial network stage.
    for epoch in range(start_epoch, epochs):
        # Train each epoch for adversarial network.
        train_adversarial(train_dataloader, epoch)
        # Verify each epoch for adversarial network.
        psnr_value, pie_value, lpips_value = validate(valid_dataloader, epoch, "adversarial")
        # Determine whether the performance of the adversarial network under epoch is the best.
        is_best_psnr = psnr_value > best_psnr_value
        is_best_lpips =lpips_value < best_lpips_value
        is_best_pie = pie_value < best_pie_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        best_lpips_value = min(lpips_value, best_lpips_value)
        best_pie_value = min(pie_value, best_pie_value)
        # Save the weight of the adversarial network under epoch. If the performance of the adversarial network
        # under epoch is the best, it will save two additional files ending with `-best.pth` in the `results` directory.
        '''
        if (epoch+1)%50==0:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir1, f"d_epoch{epoch + 1}.pth"))
            torch.save(generator.state_dict(), os.path.join(exp_dir1, f"g_epoch{epoch + 1}.pth"))
        
        if is_best_psnr:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best_psnr.pth"))
        '''
        if is_best_pie:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best_pie.pth"))
        if is_best_lpips:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best_lpips.pth"))
        # Adjust the learning rate of the adversarial model.
        d_scheduler.step()
        g_scheduler.step()

    # Save the weight of the adversarial model under the last Epoch in this stage.
    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-last.pth"))


if __name__ == "__main__":
    main()
