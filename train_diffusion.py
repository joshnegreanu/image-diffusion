import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import pandas as pd
import random
import wandb

import deepinv

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
    'max_examples': 60000,
    'image_size': 32,
    'bs': 32,
    'lr': 0.00002,
    'weight_decay': 0.000001,
    'max_epochs': 100
}

model_config = {
    'in_channels': 1,
    'out_channels': 1,
    'channels': [64, 128, 256, 512, 512, 384, 256],
    'scales': [-1, -1, -1, 1, 1, 1, 0],
    'attentions': [False, True, False, False, False, True, False],
    'time_steps': 1000
}


"""
dry_run
    Runs diffusion model through random batch to
    ensure proper dimensionality. Asserts correct shape.

    Args:
        model: torch.nn.Module diffusion model
        bs: int batch size
        image_size: int size of vocab
"""
def dry_run(model, bs, image_size, in_channels, out_channels, time_steps):
    batch = torch.randn(bs, in_channels, image_size, image_size).to(device)
    t = random.randrange(1, time_steps, 1)
    out = model(batch, t)
    assert out.shape == (bs, out_channels, image_size, image_size)
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
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    criterion = nn.MSELoss()
    # criterion = deepinv.loss.MSE(reduction='mean')

    # diffusion scheduler
    beta = torch.linspace(1e-4, 0.02, time_steps, requires_grad=False).to(device)
    alpha = 1 - beta
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

            # break up data
            batch, labels = batch

            # pick noising rate
            t = random.randrange(1, time_steps, 1)

            # run batch through diffusion
            batch = batch.to(device)
            noise = torch.randn(batch.size(), requires_grad=False).to(device)
            diffuse_batch = math.sqrt(alpha_hat[t]) * batch + math.sqrt(1 - alpha_hat[t]) * noise

            # forward pass
            noise_pred = model(diffuse_batch, t)
            # noise_pred = model(diffuse_batch, torch.ones((32,)).to(device) * t, type_t='timestep')

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

        # save model each epoch
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
    # create dataset
    # dataset = ImageDataset(
    #     dataset_name="p2pfl/MNIST",
    #     max_examples=train_config['max_examples'],
    #     image_size=train_config['image_size'],
    #     bs=train_config['bs']
    # )

    transform = transforms.Compose([
        transforms.Resize(train_config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ])
    
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=train_config['bs'],
        shuffle=True,
    )

    # create diffusion model
    model = UNet(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        channels=model_config['channels'],
        scales=model_config['scales'],
        attentions=model_config['attentions'],
        time_steps=model_config['time_steps']
    ).to(device)

    # model = deepinv.models.DiffUNet(
    #     in_channels=1,
    #     out_channels=1,
    #     pretrained=None
    # ).to(device)

    dry_run(
        model=model,
        bs=train_config['bs'],
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