import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn

# dynamically select device
if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
ConvolutionalUNet
	A U-Net architecture using convolutional layers for
	denoising images in diffusion/flow matching models.
"""
class ConvolutionalUNet(nn.Module):

	"""
	ConvolutionalUNet.__init__
		Constructs necessary internal modules for
		encoder/decoder model. Uses convolutional layers and
		skip connections for per-pixel predictions.

	Args:
		...
	"""
	def __init__(self):
		super().__init__()
            
		# encoder path
		self.enc_conv1 = self.double_conv(3, 64)
		self.enc_conv2 = self.double_conv(64, 128)
		self.enc_conv3 = self.double_conv(128, 256)
		self.enc_conv4 = self.double_conv(256, 512)
		self.enc_conv5 = self.double_conv(512, 1024)
            
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            
		# decoder path
        # takes into account concatenated skip connections
		self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
		self.dec_conv1 = self.double_conv(1024, 512)
        
		self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
		self.dec_conv2 = self.double_conv(512, 256)
        
		self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
		self.dec_conv3 = self.double_conv(256, 128)
        
		self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
		self.dec_conv4 = self.double_conv(128, 64)
        
		# output layer
		self.final_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)
		
		
	def double_conv(self, in_channels, out_channels):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
			nn.ReLU()
		)
	

	"""
	ConvolutionalUNet.forward
		Runs a forward pass of the U-Net model through
		encoder and decoder with input batch of images.
		Performs skip path concatenations.

	Args:
		x: torch.Tensor of size (B, C, L, W)

	Returns:
		torch.Tensor of size (B, C, L, W)
	"""
	def forward(self, x):
		# encoder path
		enc1 = self.enc_conv1(x) # 256x256
		enc2 = self.enc_conv2(self.maxpool(enc1)) # 128x128
		enc3 = self.enc_conv3(self.maxpool(enc2)) # 64x64
		enc4 = self.enc_conv4(self.maxpool(enc3)) # 32x32
		bottleneck = self.enc_conv5(self.maxpool(enc4)) # 16x16
		
		# decoder path with skip connections
		up1 = self.up_conv1(bottleneck)
		dec1 = self.dec_conv1(torch.cat((up1, enc4), dim=1)) # 32x32
		
		up2 = self.up_conv2(dec1)
		dec2 = self.dec_conv2(torch.cat((up2, enc3), dim=1)) # 32x32
		
		up3 = self.up_conv3(dec2)
		dec3 = self.dec_conv3(torch.cat((up3, enc2), dim=1)) # 32x32
		
		up4 = self.up_conv4(dec3)
		dec4 = self.dec_conv4(torch.cat((up4, enc1), dim=1)) # 32x32
		
		# output layer
		return self.final_conv(dec4)