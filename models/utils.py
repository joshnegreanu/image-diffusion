import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# dynamically select device
if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
PositionalEncoding
	Applies positional encoding to sequential word embeddings
	via sinusoidal encoding.
"""
class PositionalEncoding(nn.Module):
	"""
	PositionalEncoding.__init__
		Initializes encoding with proper embedding dimension.

	Args:
		embed_dim: int embedding dimensionality
	"""
	def __init__(self, embed_dim):
		super().__init__()
		self.embed_dim = embed_dim

	"""
	PositionalEncoding.forward
		Applies sinusoidal positional encoding to input.
	
	Args:
		x: torch.Tensor of size (B, N, D)

	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def forward(self, x):
		batch_size = x.shape[0]
		seq_len = x.shape[1]

		# calcualte sinusoidal encodings
		pe = torch.zeros(1, seq_len, self.embed_dim).to(x.device)
		pos = torch.arange(0, seq_len, dtype=torch.float32)
		enc = torch.exp((-math.log(10000.0)) * (torch.arange(0, self.embed_dim, step=2, dtype=torch.float32) / self.embed_dim))

		# calculate positional encoding
		prod = torch.outer(pos, enc)
		pe[0, :, 0::2] = torch.sin(prod)
		pe[0, :, 1::2] = torch.cos(prod)
		pe = pe.expand(batch_size, -1, -1)

		# apply as residual
		x = x + pe
		return x


"""
TimestepEncoding
	Generates sinusoidal positional encodings
	for diffusion time steps.
"""
class TimestepEncoding(nn.Module):
	"""
	TimestepEncoding.__init__
		Constructs sinusoidal positional encodings
		for diffusion time steps.
	
	Args:
		embed_dim: int size of embedding dimension
		time_steps: int number of diffusion time steps
	"""
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


	"""
	TimestepEncoding.forward
		Retrieves positional encoding for given
		diffusion time step.
	
	Args:
		t: torch.Tensor (B,) diffusion time steps
	
	Returns:
		torch.Tensor of size (B, D)
	"""
	def forward(self, t):
		return self.embeddings[t]


"""
MultiheadAttention
	Multi-headed attention with/without causal
	masking applied.
"""
class MultiheadAttention(nn.Module):
	"""
	MultiheadAttention.__init__
		Constructs key, query, and value matrices, and
		final linear layer.
	
	Args:
		embed_dim: int size of embedding dimension
		num_heads: int number of attention heads
	"""
	def __init__(self, embed_dim, num_heads):
		super().__init__()

		assert embed_dim % num_heads == 0
		self.embed_dim = embed_dim
		self.head_dim = int(embed_dim / num_heads)
		self.num_heads = num_heads
		
		# set up key, query, and value linear transformations
		self.q_linear = nn.Linear(embed_dim, embed_dim)
		self.k_linear = nn.Linear(embed_dim, embed_dim)
		self.v_linear = nn.Linear(embed_dim, embed_dim)

		self.concat_linear = nn.Linear(embed_dim, embed_dim)


	"""
	MultiheadAttention.scaled_dot_product_attention
		Applies scaled dot product attention to input
		previously passed through key, query, and value
		transformations.
	
	Args:
		q: torch.Tensor input queries
		k: torch.Tensor input keys
		v: torch.Tensor input values
		is_causal: boolean causal masking flag
	
	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def scaled_dot_product_attention(self, q, k, v, is_causal):
		seq_len = q.shape[1]

		# dot product self attention
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)
		dots = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

		# apply causal mask if causal
		if is_causal:
			mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
			causal_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
			dots = dots + causal_mask
		
		attn = F.softmax(dots, dim=-1)
		return torch.matmul(attn, v).transpose(1, 2).contiguous()

	
	"""
	MultiheadAttention.forward
		Runs a forward pass through multiheaded attention
		layer. Splits input dimensions across heads,
		runs through query, key, and value transformations,
		applies scaled dot product attention, concatenates
		and passes through a final linear layer.
	
	Args:
		x: torch.Tensor of size (B, N, D)
		is_causal: boolean causal masking flag
	
	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def forward(self, x, is_causal):
		bs = x.shape[0]

		# run through query, key, and value transformations
		q = self.q_linear(x).reshape(bs, -1, self.num_heads, self.head_dim)
		k = self.k_linear(x).reshape(bs, -1, self.num_heads, self.head_dim)
		v = self.v_linear(x).reshape(bs, -1, self.num_heads, self.head_dim)

		# calculate attentions, concatenate multiple heads
		attn = self.scaled_dot_product_attention(q, k, v, is_causal)
		attn = attn.reshape(bs, -1, self.embed_dim)
		return self.concat_linear(attn)


# """
# ResBlock
# 	Residual block with two convolutional layers.
# """
# class ResBlock(nn.Module):
# 	"""
# 	ResBlock.__init__
# 		Constructs two convolutional layers
# 		with ReLU activations.
	
# 	Args:
# 		num_channels: int number of input/output channels
# 	"""
# 	def __init__(self, in_channels, out_channels):
# 		super().__init__()

# 		self.block_pass = nn.Sequential(
# 			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
# 			nn.ReLU(inplace=True),
# 			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
# 			nn.ReLU(inplace=True)
# 		)
	

# 	"""
# 	ResBlock.forward
# 		Runs a forward pass through the residual block,
# 		applying timestep positional embedding.
	
# 	Args:
# 		x: torch.Tensor input feature map
# 		pos_enc: torch.Tensor positional embedding to add
	
# 	Returns:
# 		torch.Tensor of size (B, C, H, W)
# 	"""
# 	def forward(self, x):
# 		return x + self.block_pass(x)


"""
UNetLayer
	Single layer of a U-Net architecture, consisting
	of two residual blocks, optional full self-attention,
	and up-/down-sampling.
"""
class UNetLayer(nn.Module):
	def __init__(self, in_channels, out_channels, scale, attention):
		super().__init__()

		# handle scaling
		self.scale = None
		if scale == 1:
			self.scale = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
		elif scale == -1:
			self.scale = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
		
		self.relu = nn.ReLU(inplace=True)

		self.attention = None
		if attention:
			self.attention = MultiheadAttention(embed_dim=out_channels, num_heads=4)

		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)


	"""
	UNetLayer.forward
		Runs a forward pass through the U-Net layer,
		applying two residual blocks, optional full
		self-attention, and up-/down-sampling.
	
	Args:
		x: torch.Tensor input feature map
		pos_enc: torch.Tensor positional embedding to add
	
	Returns:
		torch.Tensor of size (B, C', H', W'), torch.Tensor of size (B, C, H, W)
	"""
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		
		# apply full self-attention if enabled
		if self.attention:
			h, w = x.shape[2], x.shape[3]
			x = rearrange(x, 'b c h w -> b (h w) c')
			x = self.attention(x, is_causal=False)
			x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

		x = self.conv2(x)
		x = self.relu(x)

		# add residual only if no scaling
		r = x
		if self.scale:
			x = self.scale(x)
		return x, r


"""
TransformerLayer
	Individual transformer layer employing
	multiheaded attention and a feed forward.
"""
class TransformerLayer(nn.Module):
	"""
	TransformerLayer.__init__
		Configures internal multiheaded attention
		layer, feed forward layer, and batch
		normalization.
	
	Args:
		embed_dim: int embedding dimension
		num_head: int number of heads
	"""
	def __init__(self, embed_dim, num_heads):
		super().__init__()
		# self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
		self.attn_layer = MultiheadAttention(embed_dim, num_heads)

		self.feed_forward = nn.Sequential(
			nn.Linear(embed_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, embed_dim)
		)

		self.layer_norm = nn.LayerNorm(embed_dim)
	

	"""
	TransformerLayer.forward
		Runs a forward pass through the transformer
		layer's self attention (with residual),
		batch norm, feedfoward (with resudial),
		and batch norm.
	
	Args:
		x: torch.Tensor of size (B, N, D)
		is_causal: boolean causal masking flag
	
	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def forward(self, x, is_causal):
		# run through residual attention layer (with causal mask if specified)
		x = x + self.attn_layer(x, is_causal)
		x = self.layer_norm(x)

		# run through feed forward network
		x = x + self.feed_forward(x)
		return self.layer_norm(x)


"""
Transformer
	Full multiheaded and multilayered
	transformer.
"""
class Transformer(nn.Module):
	"""
	Transformer.__init__
		Configures interal list of
		transformer layers.
	
	Args:
		embed_dim: int embedding dimension
		num_heads: int number of heads
		num_layers: int number of layers
	"""
	def __init__(self, embed_dim, num_heads, num_layers):
		super().__init__()
		# build tranformer layers
		self.transformer_layers = nn.ModuleList(
			[TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)]
		)
		
	
	"""
	Transformer.forward
		Runs a forward pass through the
		transformer.
	
	Args:
		x: torch.Tensor of size (B, N, D)
		is_causal: boolean causal masking flag
	"""
	def forward(self, x, is_causal):
		# run through layers
		for layer in self.transformer_layers:
			x = layer(x, is_causal)
		return x