import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        """Update learning rate"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    

def train_epoch(
    model, 
    dataloader, 
    optimizer, 
    criterion, 
    device, 
    grad_clip_norm=1.0, 
    grad_accum_steps=1, 
    use_amp=True
):
    """
    Train the model for one epoch using optional mixed precision and gradient accumulation.

    Args:
        model: Model to train.
        dataloader: DataLoader providing batches.
        optimizer: Optimizer instance.
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: torch.device for computations.
        grad_clip_norm (float): Maximum norm for gradient clipping.
        grad_accum_steps (int): Number of steps for gradient accumulation.
        use_amp (bool): Whether to use automatic mixed precision (AMP).

    Returns:
        tuple:
            - avg_loss (float): Average training loss.
            - char_acc (float): Training character-level accuracy.
    """
    model.train()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixed precision forward pass
        if use_amp and device.type == 'cuda':
            with autocast():
                logits = model(images)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
        else:
            logits = model(images)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
        
        loss = loss / grad_accum_steps
        
        # Backward pass with gradient scaling
        if use_amp and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step with gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        pad_idx = criterion.ignore_index
        mask = labels != pad_idx
        correct_chars += ((predictions == labels) & mask).sum().item()
        total_chars += mask.sum().item()
        
        total_loss += loss.item() * grad_accum_steps
        
        pbar.set_postfix({
            'loss': f'{loss.item() * grad_accum_steps:.4f}',
            'char_acc': f'{correct_chars / total_chars:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    char_acc = correct_chars / total_chars
    
    return avg_loss, char_acc
