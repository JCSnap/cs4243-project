"""
Main training script for ViTSTR CAPTCHA Recognition
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from models import ViTSTR
from data import CAPTCHATokenizer, create_data_loaders
from utils import (
    train_epoch, 
    evaluate, 
    save_checkpoint, 
    save_best_model
)

def load_data_from_directory(data_dir):
    """
    Load image file paths and labels from a specified directory.
    Assumes filenames encode labels (before dash or extension).

    Args:
        data_dir (str): Directory containing images.

    Returns:
        tuple: (image_paths (list of str), labels (list of str))
    """
    image_paths = []
    labels = []
    
    for img_file in Path(data_dir).glob('*.png'):
        # Extract label from filename (customize based on your naming convention)
        filename = img_file.stem
        if '-' in filename:
            label = filename.split('-')[0]
        else:
            label = filename
        image_paths.append(str(img_file))
        labels.append(label)
    
    return image_paths, labels


def main():
    """
    Main routine for model training:
    - Loads data and splits into train/val.
    - Sets up model, optimizer, loss, and training loop.
    - Logs key information and metrics.
    - Saves checkpoints and best model.
    """
    torch.backends.cudnn.benchmark = True

    # Configuration
    cfg = Config()
    
    # Set device
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    Path(cfg.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = CAPTCHATokenizer(chars=cfg.CHARS)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load data
    print("Loading data...")
    image_paths, labels = load_data_from_directory(cfg.DATA_DIR)
    
    # Split into train and validation
    split_idx = int(len(image_paths) * cfg.TRAIN_SPLIT)
    train_paths = image_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_paths = image_paths[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels,
        tokenizer, cfg.BATCH_SIZE, cfg.NUM_WORKERS,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH
    )
    
    # Initialize model
    model = ViTSTR(
        img_height=cfg.IMG_HEIGHT,
        img_width=cfg.IMG_WIDTH,
        patch_size=cfg.PATCH_SIZE,
        in_channels=cfg.IN_CHANNELS,
        num_classes=tokenizer.vocab_size,
        max_seq_len=cfg.MAX_SEQ_LEN,
        embed_dim=cfg.EMBED_DIM,
        depth=cfg.DEPTH,
        num_heads=cfg.NUM_HEADS,
        mlp_ratio=cfg.MLP_RATIO,
        dropout=cfg.DROPOUT
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.PAD_IDX,
        label_smoothing=cfg.LABEL_SMOOTHING
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS}")
        print("-" * 80)
        
        # Update learning rate
        # current_lr = scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_char_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg.GRAD_CLIP_NORM, cfg.GRAD_ACCUM_STEPS
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Char Acc: {train_char_acc:.4f}")
        
        # Validate
        val_loss, val_char_acc, val_seq_acc = evaluate(
            model, val_loader, criterion, device, tokenizer
        )
        
        print(f"Val Loss: {val_loss:.4f} | Val Char Acc: {val_char_acc:.4f} | Val Seq Acc: {val_seq_acc:.4f}")
        
        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'train_char_acc': train_char_acc,
            'val_loss': val_loss,
            'val_char_acc': val_char_acc,
            'val_seq_acc': val_seq_acc,
            'learning_rate': current_lr
        }
        
        checkpoint_path = os.path.join(
            cfg.CHECKPOINT_DIR, 
            f'checkpoint_epoch_optimised_{epoch+1}.pth'
        )
        save_checkpoint(model, optimizer, epoch + 1, metrics, checkpoint_path)
        
        # Save best model
        if val_seq_acc > best_val_acc:
            best_val_acc = val_seq_acc
            save_best_model(model, metrics, cfg.CHECKPOINT_DIR)
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation sequence accuracy: {best_val_acc:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
