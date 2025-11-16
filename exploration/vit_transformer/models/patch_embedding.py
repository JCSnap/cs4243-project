"""
Patch Embedding Layer for Vision Transformer
Converts images into sequences of patch embeddings
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    PyTorch layer for converting images into sequential patch embeddings via Conv2d, 
    as required for ViT and related transformer models.

    Attributes:
        img_height (int): Input image height.
        img_width (int): Input image width.
        patch_size (int): Size of square patch.
        num_patches_h (int): Patches along image height.
        num_patches_w (int): Patches along image width.
        n_patches (int): Total number of patches.
        proj (nn.Conv2d): Convolutional projection (patching + embedding).
    """
    
    def __init__(
        self,
        img_height=32,
        img_width=128,
        patch_size=4, 
        in_channels=3,
        embed_dim=768
    ):
        """
        Initializes parameters and projection layer for patch extraction and embedding.

        Args:
            img_height (int): Input image height.
            img_width (int): Input image width.
            patch_size (int): Patch size.
            in_channels (int): Number of channels in the input image.
            embed_dim (int): Feature dimension for each patch embedding.
        """
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        
        # Calculate number of patches
        self.num_patches_h = img_height // patch_size
        self.num_patches_w = img_width // patch_size
        self.n_patches = self.num_patches_h * self.num_patches_w
        
        # Use Conv2d as patch extraction + linear projection
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Converts an image batch into a sequence of patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, n_patches, embed_dim).
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x
