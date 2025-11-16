"""
Model checkpoint utilities
"""

import torch
import os
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    Save model and optimizer state along with training metrics.

    Args:
        model: Model to save.
        optimizer: Optimizer instance.
        epoch (int): Current epoch number.
        metrics (dict): Dictionary of metric values.
        filepath (str): Path to save the checkpoint file.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model and optimizer state from checkpoint.

    Args:
        model: Model to load state into.
        optimizer: Optimizer to load state into.
        filepath (str): Path to the checkpoint file.
        device: torch device mapping for loading.

    Returns:
        tuple: (epoch (int), metrics (dict))
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    
    return epoch, metrics


def save_best_model(model, metrics, checkpoint_dir, filename='best_model.pth'):
    """
    Save model state dict and evaluation metrics for the best performing model.

    Args:
        model: Model to save.
        metrics (dict): Metrics of best model.
        checkpoint_dir (str): Directory to save model.
        filename (str): File name for best model (default is 'best_model.pth').
    """
    filepath = os.path.join(checkpoint_dir, filename)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, filepath)
    
    print(f"Best model saved with val_seq_acc: {metrics['val_seq_acc']:.4f}")
