"""
Complete ViTSTR Model for CAPTCHA Recognition
"""

import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .transformer_block import TransformerBlock

class ViTSTR(nn.Module):
    """
    Vision Transformer for Scene Text Recognition, customized for CAPTCHA decoding.

    Attributes:
        num_classes (int): Number of output classes.
        max_seq_len (int): Maximum output length.
        embed_dim (int): Embedding dimension per patch.
        patch_embed (PatchEmbedding): Patch embedding module.
        pos_embed (nn.Parameter): Learnable positional encoding.
        pos_drop (nn.Dropout): Dropout for regularization.
        blocks (nn.ModuleList): Transformer encoder layers.
        norm (nn.LayerNorm): Final normalization.
        head (nn.Linear): Output classifier for character predictions.
    """
    
    def __init__(
        self, 
        img_height=32,
        img_width=128,
        patch_size=4,
        in_channels=3,
        num_classes=63,
        max_seq_len=15,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        """
        Builds the complete ViTSTR model.

        Args:
            img_height (int): Input image height.
            img_width (int): Input image width.
            patch_size (int): Size of square patches.
            in_channels (int): Channels in input images.
            num_classes (int): Number of possible output classes.
            max_seq_len (int): Maximum output sequence length.
            embed_dim (int): Patch embedding dimension.
            depth (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio for MLP expansion.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_height, img_width, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.n_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Character prediction head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """
        Custom weight initialization for layers in the model.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for ViTSTR.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Character logits for each output position,
                shape (B, max_seq_len, num_classes).
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Take first max_seq_len patches for character prediction
        seq_length = min(self.max_seq_len, x.size(1))
        x = x[:, :seq_length, :]
        
        # Predict characters for each position
        logits = self.head(x)  # (B, max_seq_len, num_classes)
        
        return logits
    
    def get_num_params(self):
        """
        Returns the total number of model parameters.

        Returns:
            int: Total parameter count.
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """
        Returns the total number of trainable (requires_grad=True) parameters in the model.

        Returns:
            int: Count of all parameters in the model that will be updated during training.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
