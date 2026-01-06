import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn

from einops import rearrange

from models.utils import Transformer, PositionalEncoding

# dynamically select device
if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class VisionTransformer(nn.Module):
    def __init__(self, patch_size, in_channels, out_channels, embed_dim, num_layers, num_heads):
        super().__init__()
        
        # patch embedding layers
        self.patch_in = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.patch_out = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.patch_size = patch_size

        # positional encodings
        self.pos_enc = PositionalEncoding(embed_dim=embed_dim)

        # internal transformer
        self.transformer = Transformer(embed_dim, num_heads, num_layers)
    
    def forward(self, x):
        b, c, w, h = x.shape
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # reshape image into patches and encode
        x = rearrange(
            x,
            'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
            ph=self.patch_size,
            pw=self.patch_size,
            nh=num_patches_h,
            nw=num_patches_w
        )
        x = self.patch_in(x)
        x = self.pos_enc(x)

        # transformer pass
        x = self.transformer(x, is_causal=False)

        # decode patches and reshape back to image
        x = self.patch_out(x)
        x = rearrange(
            x,
            'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            nh=num_patches_h,
            nw=num_patches_w
        )
        return x