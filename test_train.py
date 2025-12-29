import torch
import deepinv
from torchvision import datasets, transforms

from datetime import datetime
import wandb
import os

device = "cuda"
batch_size = 32
image_size = 32

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

lr = 1e-4
epochs = 100

model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = deepinv.loss.MSE(reduction='mean')

beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# set up wandb and checkpoint path
now = datetime.now()
project_name = "diffusion-image-model"
run_name = "dim-" + now.strftime("%Y_%m_%d_%H_%m")
wandb.login()
wandb.init(project=project_name, name=run_name, config=None)
os.makedirs(f"./checkpoints/{project_name}/{run_name}", exist_ok=True)

iteration = 0
for epoch in range(epochs):
    model.train()
    for data, _ in train_loader:
        imgs = data.to(device)
        noise = torch.randn_like(imgs)
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

        noised_imgs = (
            sqrt_alphas_cumprod[t, None, None, None] * imgs
            + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        optimizer.zero_grad()
        estimated_noise = model(noised_imgs, t, type_t="timestep")
        loss = mse(estimated_noise, noise)
        wandb.log({"loss": loss.item()}, step=iteration)
        iteration += 1
        loss.backward()
        optimizer.step()

    torch.save(
        model.state_dict(),
        f"trained_diffusion_model{epoch}.pth",
    )