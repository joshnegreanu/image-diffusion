import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn

from einops import rearrange

# dynamically select device
if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class PositionalEncoding(nn.Module):
	def __init__(self, embed_dim, time_steps):
		super().__init__()
		self.embed_dim = embed_dim
		self.time_steps = time_steps

		positions = torch.arange(self.time_steps).unsqueeze(1).float()
		div = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / self.embed_dim))
		embeddings = torch.zeros(self.time_steps, self.embed_dim, requires_grad=False)
		embeddings[:, 0::2] = torch.sin(positions * div)
		embeddings[:, 1::2] = torch.cos(positions * div)
		self.embeddings = embeddings.to(device)

	def forward(self, t):
		return self.embeddings[t]


# implement attention better...
class Attention(nn.Module):
	def __init__(self, num_channels, num_heads):
		super().__init__()
		self.proj1 = nn.Linear(num_channels, num_channels*3)
		self.proj2 = nn.Linear(num_channels, num_channels)
		self.num_heads = num_heads

	def forward(self, x):
		h, w = x.shape[2:]
		x = rearrange(x, 'b c h w -> b (h w) c')
		x = self.proj1(x)
		x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
		q,k,v = x[0], x[1], x[2]
		x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
		x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
		x = self.proj2(x)
		return rearrange(x, 'b h w C -> b C h w')


class ResBlock(nn.Module):
	def __init__(self, num_channels):
		super().__init__()

		self.block_pass = nn.Sequential(
			# nn.GroupNorm(num_groups=32, num_channels=num_channels),
			nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			# nn.GroupNorm(num_groups=32, num_channels=num_channels),
			nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, x, pos_emb):
		x = x + pos_emb[None, :, None, None]
		return x + self.block_pass(x)


class UNetLayer(nn.Module):
	def __init__(self, num_channels, scale, attention):
		super().__init__()
		self.res_block1 = ResBlock(num_channels=num_channels)
		self.res_block2 = ResBlock(num_channels=num_channels)

		self.attention = None
		if attention:
			self.attention = Attention(num_channels=num_channels, num_heads=4)

		if scale == 1:
			self.scale = nn.ConvTranspose2d(num_channels, num_channels//2, kernel_size=4, stride=2, padding=1)
		elif scale == -1:
			self.scale = nn.Conv2d(num_channels, num_channels*2, kernel_size=3, stride=2, padding=1)
		else:
			self.scale = nn.Identity()


	def forward(self, x, pos_emb):
		x = self.res_block1(x, pos_emb)
		
		if self.attention:
			x = self.attention(x)

		x = self.res_block2(x, pos_emb)
		return self.scale(x), x


class UNet(nn.Module):
	def __init__(self, in_channels, out_channels, channels, attentions, scales, time_steps):
		super().__init__()
		self.num_layers = len(channels[:-1])

		self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
		self.conv_out = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
		self.relu = nn.ReLU()

		for i in range(self.num_layers):
			pos_enc = PositionalEncoding(
				embed_dim=channels[i],
				time_steps=time_steps
			)
			layer = UNetLayer(
				scale=scales[i],
				num_channels=channels[i],
				attention=attentions[i]
			)
			setattr(self, f'pos_enc{i+1}', pos_enc)
			setattr(self, f'layer{i+1}', layer)
		
		self.final_pos_enc = PositionalEncoding(embed_dim=channels[-1], time_steps=time_steps)
		self.final_layer = UNetLayer(scale=scales[-1], num_channels=channels[-1], attention=attentions[i])
	
	def forward(self, x, t):
		x = self.conv_in(x)

		residuals = []
		for i in range(self.num_layers // 2):
			pos_enc = getattr(self, f'pos_enc{i+1}')
			layer = getattr(self, f'layer{i+1}')
			pos_emb = pos_enc(t)
			x, y = layer(x, pos_emb)
			residuals.append(y)

		for i in range(self.num_layers // 2, self.num_layers):
			pos_enc = getattr(self, f'pos_enc{i+1}')
			layer = getattr(self, f'layer{i+1}')
			pos_emb = pos_enc(t)
			x = layer(x, pos_emb)[0]
			y = residuals[self.num_layers - i - 1]
			x = torch.concat((x, y), dim=1)
		
		pos_emb = self.final_pos_enc(t)
		x = self.final_layer(x, pos_emb)[0]
		x = self.relu(x)
		return self.conv_out(x)