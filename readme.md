# image-diffusion

This repo is a playground for me to explore image generation methods such as denoising diffusion and flow-matching models.

## Project Structure

Below is a repo folder structure diagram:
```
language-diffusion/
├── data/
...
├── models/
│   └── UNet.py
│   └── utils.py
...
└── README.md
```

## Models

A CNN-based UNet for per-pixel regression and time-step positional encoding. Used to predict added noise in denoising diffusion. The model is implemented under [`models/UNet.py`](./models/UNet.py), using sub-modules implemented in [`models/utils.py`](./models/utils.py).

The model can be initialized with the following arguments:
```python
model_config = {
    'in_channels': 3,
    'out_channels': 3,
    'channels': [64, 128, 256, 512, 512, 384, 256],
    'scales': [-1, -1, -1, 1, 1, 1, 0],
    'attentions': [False, True, False, False, False, True, False],
    'time_steps': 100
}

model = UNet(
    in_channels=model_config['in_channels'],
    out_channels=model_config['out_channels'],
    channels=model_config['channels'],
    scales=model_config['scales'],
    attentions=model_config['attentions'],
    time_steps=model_config['time_steps']
).to(device)
```

## Denoising Diffusion Image Generation

...

## Flow-Matching Image Generation

...
