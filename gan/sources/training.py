#!/usr/bin python3

import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sources.generator import Generator
from sources.discriminator import Discriminator
from sources.plotting import plot_loss, plot_real_fake
from sources.notify import notifier

seed = 657587
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

def weight_clipping(model: nn.Module, clip_value: float) -> None:
    for param in model.parameters():
        param.data.clamp_(-clip_value, clip_value)

def check_gradients_norm(model: nn.Module) -> float:
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def add_instance_noise(images: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Add Gaussian noise to images
    See: https://www.infereconfig.nce.vc/instance-noise-a-trick-for-stabilising-gan-training/
    """
    if noise_std == 0:
        return images
    return images + torch.randn_like(images) * noise_std

def wasserstein_loss(real_output: torch.Tensor, fake_output: torch.Tensor) -> torch.Tensor:
    return -torch.mean(real_output) + torch.mean(fake_output)

def gradient_penalty(Dnet, real_samples, fake_samples, device) -> torch.Tensor:
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = Dnet(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

def training(device, config):
    notifier = notifier()
    dataset = datasets.ImageFolder(root=config.dataroot,
                                   transform=transforms.Compose([
                                           transforms.Resize(config.image_size),
                                           transforms.CenterCrop(config.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    real_batch = next(iter(dataloader))
    # generator init
    netG = Generator(config).to(device)
    if (device.type == 'cuda') and (config.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(config.ngpu)))
    print(netG)
    # discriminator init
    netD = Discriminator(config).to(device)
    if (device.type == 'cuda') and (config.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(config.ngpu)))
    print(netD)

    real_label = 1
    # optimizer
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr_D, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_G, betas=(config.beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    current_noise_std = config.initial_noise_std
    print("start training...")
    for epoch in range(config.num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            ## Discriminator real batch training, for first gradient update ##

            netD.zero_grad()
            # Train with real batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            noisy_real = add_instance_noise(real_cpu, current_noise_std)
            real_output = netD(noisy_real).view(-1)
            # Train with fake batch
            noise = torch.randn(b_size, config.nz, 1, 1, device=device)  # Generate batch of latents
            fakes = netG(noise)
            noisy_fakes = add_instance_noise(fakes.detach(), current_noise_std)
            fake_output = netD(noisy_fakes).view(-1)
            # Compute gradient penalty
            gp = gradient_penalty(netD, noisy_real, noisy_fakes, device)  # Add gradient penalty
            # Compute Wasserstein loss
            lossD = -real_output.mean() + fake_output.mean() + 10 * gp  # WGAN loss with gp
            # Backward pass and step
            lossD.backward()
            optimizerD.step()

            if lossD.item() == 0.0:
                print("Discriminator loss is 0.0, failure!")
                exit(1)

            ############################
            # Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            # Add noise to fake images for generator update
            noisy_fakes = add_instance_noise(fakes, current_noise_std)
            output = netD(noisy_fakes).view(-1)
            #calculate generator loss
            lossG = wasserstein_loss(output, label)
            # calculate generator gradients in backward
            lossG.backward()
            D_G_z2 = output.mean().item()
            # gradient step
            optimizerG.step()

            current_noise_std *= config.noise_decay_rate
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch, config.num_epochs, i, len(dataloader), lossD.item(), lossG.item()))
            # Save Losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            if (iters % 500 == 0) or ((epoch == config.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fakes = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fakes, padding=2, normalize=True))
            iters += 1
        torch.save(netD.state_dict(), f"{config.saveroot}/checkpoints/checkpoint_D_{epoch}.pt")
        torch.save(netG.state_dict(), f"{config.saveroot}/checkpoints/checkpoint_G_{epoch}.pt")
        if epoch > 0:
            os.remove(f"{config.saveroot}/checkpoints/checkpoint_D_{epoch-1}.pt")
            os.remove(f"{config.saveroot}/checkpoints/checkpoint_G_{epoch-1}.pt")

    torch.save(netD, f"{config.saveroot}/model_D.pt")
    torch.save(netG, f"{config.saveroot}/model_G.pt")

    real_batch = next(iter(dataloader))
    plot_loss(G_losses, D_losses)
    plot_real_fake(real_batch, img_list, device)

    notifier.notify_phone("GAN training done", f"Loss_G: {lossG.item()} Loss_D: {lossD.item()}")
