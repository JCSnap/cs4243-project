"""
Transformer Encoder Block
Combines Multi-Head Attention with Feed-Forward Network
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class MLP(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        """
        Position-wise feed-forward network used in transformer blocks.

        Args:
            embed_dim (int): Input and output feature dimension.
            mlp_ratio (float): Ratio for hidden layer width.
            dropout (float): Dropout rate.
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for the MLP block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm architecture"""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        """
        Transformer encoder block with pre-normalization, multi-head attention, residual connections,
        and feed-forward network.

        Args:
            embed_dim (int): Embedding dimension for features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio for hidden layer in MLP.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        """"
        Forward pass for transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)

        Returns:
            torch.Tensor: Output sequence of shape (B, N, C)
        """
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
