import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import pandas as pd
import random
import wandb

import sys
import signal
from functools import partial
import torchvision.transforms as transforms

from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms

from datetime import datetime
from tqdm import tqdm

from models.UNet import UNet
from models.VisionTransformer import VisionTransformer
from data.ImageDataset import ImageDataset

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
Training and model configurations.
Can be changed prior to training.
"""
train_config = {
    'image_size': 128,
    'batch_size': 32,
    'learning_rate': 0.00002,
    'weight_decay': 0.000001,
    'max_epochs': 500,
    'save_freq': 100
}

model_config = {
    'in_channels': 3,
    'out_channels': 3,
    'num_layers': 6,
    'time_steps': 1000
}

# model_config = {
#     'patch_size': 4,
#     'in_channels': 1,
#     'out_channels': 1,
#     'embed_dim': 256,
#     'num_layers': 8,
#     'num_heads': 8,
#     'time_steps': 100
# }


"""
dry_run
    Runs diffusion model through random batch to
    ensure proper dimensionality. Asserts correct shape.

    Args:
        model: torch.nn.Module diffusion model
        batch_size: int batch size
        image_size: int size of vocab
"""
def dry_run(model, batch_size, image_size, in_channels, out_channels, time_steps):
    batch = torch.randn(batch_size, in_channels, image_size, image_size).to(device)
    t = torch.randint(0, time_steps, (batch_size,)).to(device)
    out = model(batch, t)
    assert out.shape == (batch_size, out_channels, image_size, image_size)
    print("[dry_run] passed")


"""
interrupt_handler
    Save model checkpoint in case of terminal interrupt.
    Special checkpoint file tag not to override current
    epoch checkpoint.

Args:
    A ton haha...
"""
def interrupt_handler(
    epoch,
    loss,
    model,
    scheduler,
    optimizer,
    train_config,
    model_config,
    project_name,
    run_name,
    sig,
    frame
):
    # save model each epoch
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_config': train_config,
            'model_config': model_config},
            f"./checkpoints/{project_name}/{run_name}/{run_name}_epoch{epoch}_int.pth"
        )


"""
train
    Sets up wandb logging, creates AdamW optimizer,
    custom linear warmup and cosine annealing scheduler,
    trains for designated number of epochs, saving model
    checkpoints each epoch.

    Args:
        model: torch.nn.Module diffusion model
        data_loader: torch.DataLoader training data
"""
def train(model, dataloader, time_steps):
    # set up wandb and checkpoint path
    now = datetime.now()
    project_name = "diffusion-image-model"
    run_name = "dim-" + now.strftime("%Y_%m_%d_%H_%m")
    wandb.login()
    wandb.init(project=project_name, name=run_name, config=train_config)
    os.makedirs(f"./checkpoints/{project_name}/{run_name}", exist_ok=True)

    # optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
    criterion = nn.MSELoss()

    # diffusion scheduler
    beta = torch.linspace(1e-4, 6e-2, time_steps, requires_grad=False).to(device)
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0).requires_grad_(False).to(device)

    # construct linear warmup and cosine annealing cooldown
    warmup_epochs = int(train_config['max_epochs'] / 10)
    cooldown_epochs = train_config['max_epochs'] - warmup_epochs
    epoch_len = len(dataloader)

    linear = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_epochs*epoch_len)
    cosine = CosineAnnealingLR(optimizer, T_max=cooldown_epochs*epoch_len, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs*epoch_len])

    model.train()

    # main training loop
    pbar = tqdm(total=(train_config['max_epochs'])*epoch_len, desc="Training Iterations", unit="batch")
    iteration = 0
    for epoch in range(train_config['max_epochs']):
        # signal catching to save model on interrupt
        signal.signal(signal.SIGINT, partial(interrupt_handler,
            epoch, None,
            scheduler, optimizer,
            train_config, model_config,
            project_name, run_name))
        signal.signal(signal.SIGTERM, partial(interrupt_handler,
            epoch, None,
            scheduler, optimizer,
            train_config, model_config,
            project_name, run_name))

        # minibatch gradient descent
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            wandb.log({'learning-rate': scheduler.get_last_lr()[0]}, step=iteration)

            # pick out batch
            batch, labels = batch
            batch = batch.to(device)

            # random sample batched noising rate
            t = torch.randint(0, time_steps, (batch.size(0),), requires_grad=False).to(device).contiguous()

            # run batch through diffusion
            noise = torch.randn(batch.size(), requires_grad=False).to(device)
            diffuse_batch = (alpha_hat[t].sqrt().reshape(-1, 1, 1, 1) * batch) + ((1 - alpha_hat[t]).sqrt().reshape(-1, 1, 1, 1) * noise)

            # forward pass
            noise_pred = model(diffuse_batch, t)

            # compute L2 loss between predicted noise and true noise
            loss = criterion(noise_pred, noise)
            epoch_loss += loss
            wandb.log({"loss": loss.item()}, step=iteration)

            # optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            iteration += 1
            scheduler.step()

        # save model per save frequency
        if (epoch + 1) % train_config['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'loss': epoch_loss / epoch_len,
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_config': train_config,
                'model_config': model_config},
                f"./checkpoints/{project_name}/{run_name}/{run_name}_epoch{epoch}_end.pth"
            )

    wandb.finish()
    pbar.close()


"""
main
    Builds a model, checks through a dry run, runs
    through training cycle.
"""
def main():
    # image transformations
    transform = transforms.Compose([
        transforms.Resize((train_config['image_size'], train_config['image_size'])),
        transforms.ToTensor(),
        # normalize to [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if model_config['in_channels'] == 3 else transforms.Normalize((0.5,), (0.5,))
    ])
    
    # load stanford cars dataset
    dataloader = torch.utils.data.DataLoader(
        datasets.StanfordCars(root="./data", split='train', download=False, transform=transform),
        batch_size=train_config['batch_size'],
        shuffle=True,
    )

    # create UNet diffusion model
    model = UNet(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        num_layers=model_config['num_layers'],
        time_steps=model_config['time_steps'],
    ).to(device)

    """
    # create VisionTransformer diffusion model (NOT WORKING CURRENTLY)
    model = VisionTransformer(
        patch_size=model_config['patch_size'],
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        embed_dim=model_config['embed_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        time_steps=model_config['time_steps']
    ).to(device)
    """

    # dry run to ensure proper dimensionality
    dry_run(
        model=model,
        bs=train_config['batch_size'],
        image_size=train_config['image_size'],
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        time_steps=model_config['time_steps']
    )

    # enter training cycle
    train(
        model=model,
        dataloader=dataloader,
        time_steps=model_config['time_steps']
    )


if __name__ == '__main__':
    main()