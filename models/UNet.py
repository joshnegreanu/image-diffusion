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
	def __init__(self, in_channels, out_channels, num_layers, time_steps):
		super().__init__()
		# auto-calculate encoder/decoder layers
		down_channels = []
		down_attns = []
		for i in range(num_layers):
			down_channels.append(2**(6+i))
			down_attns.append(False)
		down_attns[-1] = True

		up_channels = down_channels[::-1]
		up_attns = down_attns[::-1]

		# encoding/decoding bookkeeping
		self.num_down = len(down_channels)
		self.num_up = len(up_channels)

		# get input into higher channel space
		self.conv_in = nn.Conv2d(in_channels, down_channels[0]//2, kernel_size=3, padding=1)
		self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, padding=1)

		# build down layers dynamically
		for i in range(self.num_down):
			layer = UNetLayer(
				in_channels=down_channels[i]//2,
				out_channels=down_channels[i],
				scale=-1,
				attention=down_attns[i],
				time_steps=time_steps
			)
			setattr(self, f'down_layer{i+1}', layer)
		
		# bottleneck layer
		self.bottleneck_layer = UNetLayer(
			in_channels=down_channels[-1],
			out_channels=up_channels[0],
			scale=1,
			attention=up_attns[0],
			time_steps=time_steps
		)

		# build up layers dynamically
		for i in range(self.num_up-1):
			layer = UNetLayer(
				in_channels=up_channels[i]*2,
				out_channels=up_channels[i]//2,
				scale=1,
				attention=up_attns[i],
				time_steps=time_steps
			)
			setattr(self, f'up_layer{i+1}', layer)

		# final layer
		self.final_layer = UNetLayer(
			in_channels=up_channels[-1]*2,
			out_channels=up_channels[-1],
			scale=0,
			attention=up_attns[-1],
			time_steps=time_steps
		)
	

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
		for i in range(self.num_down):
			layer = getattr(self, f'down_layer{i+1}')
			x, r = layer(x, t)
			residuals.append(r)
		
		x = self.bottleneck_layer(x, t)[0]
		r = residuals[-1]
		x = torch.concat((x, r), dim=1)

		# upsampling path (with skip connections)
		for i in range(self.num_up-1):
			layer = getattr(self, f'up_layer{i+1}')
			x = layer(x, t)[0]
			r = residuals[self.num_down-2-i]
			x = torch.concat((x, r), dim=1)

		# final layer
		x = self.final_layer(x, t)[0]
		return self.conv_out(x)