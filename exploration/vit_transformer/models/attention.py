"""
Multi-Head Self-Attention mechanism
"""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with scaled dot-product"""
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        """
        Multi-head self-attention mechanism with scaled dot-product attention.

        Args:
            embed_dim (int): Input and output feature dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention weights and projection.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for all heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Perform multi-head self-attention on input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
                               B=batch size, N=sequence length, C=embed_dim

        Returns:
            torch.Tensor: Attended output of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention: (Q @ K^T) / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
