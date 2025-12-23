import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd
import random
import spacy

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms

from datasets import load_dataset
from tqdm import tqdm

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
ImageDataset
    Custom dataset class for storing image dataset
    and custom dataloader.
"""
class ImageDataset(Dataset):
    def __init__(self, dataset_name, max_examples, image_size, bs):
        """
        ImageDataset.__init__
            Constructs internal dataset for training. Loads
            dataset from huggingface datasets library, samples
            from dataset, downscales images.
        
        Args:
            dataset_name: string huggingface dataset name
            max_examples: int maximum number of training examples
            max_dim: int image dimension (height/width)
        """
        dataset = load_dataset(dataset_name, split='train')

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor()
        ])
        self.data = [transform(x['image']) for x in random.sample(list(dataset), max_examples)]
        print("[dataset] loaded")

        self.image_size = image_size
        self.bs = bs


    """
    ImageDataset.__len__
        Provides length of dataset.
    
    Returns:
        int number of training examples
    """
    def __len__(self):
        return len(self.data)


    """
    ImageDataset.__getitem__
        Returns a single training example pertaining
        to a given index.

    Returns:
        torch.Tensor 
    """
    def __getitem__(self, idx):
        # normalize image to [0, 1]
        img = self.data[idx]
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        return img


    """
    ImageDataset.create_dataloader
        Creates a dataloader for internal image
        dataset.
    
    Returns:
        torch.utils.data.DataLoader for custom dataset
    """
    def create_dataloader(self):
        return DataLoader(self, batch_size=self.bs, num_workers=4, shuffle=True, drop_last=True)