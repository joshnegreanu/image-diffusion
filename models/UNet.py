import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn

from einops import rearrange

from models.utils import TimestepEncoding, UNetLayer

# dynamically select device
if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
UNet
	Implementation of a convolutional U-Net
	architecture with attention layers.
"""
class UNet(nn.Module):
	"""
	UNet.__init__
		Constructs a convolutional U-Net
		architecture with attention layers.
	
	Args:
		in_channels: int number of input channels
		out_channels: int number of output channels
		channels: list of int number of channels per layer
		attentions: list of bool whether to use attention per layer
		scales: list of int scaling factor per layer
		time_steps: int number of diffusion time steps
	"""
	def __init__(self, in_channels, out_channels, channels, attentions, scales, time_steps):
		super().__init__()
		self.num_layers = len(channels[:-1])

		self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
		self.conv_out = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
		self.relu = nn.ReLU()

		# build layers dynamically
		for i in range(self.num_layers):
			pos_enc = TimestepEncoding(
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
		
		self.final_pos_enc = TimestepEncoding(embed_dim=channels[-1], time_steps=time_steps)
		self.final_layer = UNetLayer(scale=scales[-1], num_channels=channels[-1], attention=attentions[i])
	

	"""
	UNet.forward
		Runs a forward pass through the U-Net,
		applying skip connections.
	
	Args:
		x: torch.Tensor input feature map
		t: torch.Tensor of size (B,) diffusion time steps
	
	Returns:
		torch.Tensor of size (B, out_channels, H, W)
	"""
	def forward(self, x, t):
		x = self.conv_in(x)

		residuals = []
		# downsampling path
		for i in range(self.num_layers // 2):
			pos_enc = getattr(self, f'pos_enc{i+1}')
			layer = getattr(self, f'layer{i+1}')
			pos_emb = pos_enc(t)
			x, y = layer(x, pos_emb)
			residuals.append(y)

		# upsampling path (with skip connections)
		for i in range(self.num_layers // 2, self.num_layers):
			pos_enc = getattr(self, f'pos_enc{i+1}')
			layer = getattr(self, f'layer{i+1}')
			pos_emb = pos_enc(t)
			x = layer(x, pos_emb)[0]
			y = residuals[self.num_layers - i - 1]
			x = torch.concat((x, y), dim=1)
		
		# final layer
		pos_emb = self.final_pos_enc(t)
		x = self.final_layer(x, pos_emb)[0]
		x = self.relu(x)
		return self.conv_out(x)