# ViTSTR CAPTCHA Recognition

This repository implements a Vision Transformer for Scene Text Recognition (ViTSTR) tailored to CAPTCHA decoding, including dataset utilities, tokenizer, model, training loop, evaluation metrics, checkpointing, and an inference script for single-image prediction.

### Highlights

- Pure PyTorch implementation of a compact ViTSTR with Conv2d patch embedding, learnable positional encodings, pre-norm Transformer encoder blocks, and a per-position character classifier head.
- Simple character tokenizer with a configurable vocabulary and explicit [PAD] handling for fixed-length targets.
- Training loop with label smoothing, gradient clipping, optional AMP, character-level and sequence-level accuracy metrics, and best-checkpoint saving.
- Inference script that loads the best model checkpoint and predicts text for a given image path.

### Repository structure

- config.py — central configuration for model, data, training, paths, and device.
- dataset.py — CAPTCHADataset, preprocessing (resize+pad+normalize), and DataLoader builders.
- tokenizer.py — CAPTCHATokenizer for encode/decode with [PAD] and batch decoding.
- vitstr_model.py — ViTSTR architecture with patch embedding, positional encodings, Transformer blocks, and classifier head.
- patch_embedding.py — Conv2d-based patch embedder used by the ViT model.
- attention.py — multi-head self-attention module with scaled dot-product.
- transformer_block.py — encoder block with pre-norm, MHA, and MLP.
- training.py — train_epoch and WarmupCosineScheduler utilities.
- evaluation.py — evaluation loop, example predictions, and batch prediction helper.
- checkpoint.py — save/load checkpoint utilities and best model saver
- train.py — end-to-end training script with directory-based data loading and splitting.
- inference.py — single-image inference loader using best_model.pth.
- requirements.txt — Python dependencies.

Note on imports: train.py and inference.py expect packages/models named models, data, and utils that re-export classes and functions from these files; either place files under packages with **init**.py or adjust imports to direct module paths.

## Installation

- Python 3.9+ is recommended, and PyTorch 2.0+ is required by the provided requirements.
- Install dependencies:

```bash
pip install -r requirements.txt
```

This installs torch, torchvision, numpy, pillow, scipy, matplotlib, tqdm, and tensorboard versions specified in requirements.txt.

## Data preparation

- Place your CAPTCHA images under the directory configured as DATA_DIR in config.py (default: ./data/captchas).
- By default, train.py derives labels from filenames; it uses text before the first dash if present, otherwise the entire stem, so adapt naming or edit load_data_from_directory accordingly.
- The dataset resizes each image to IMG_WIDTH × IMG_HEIGHT with preserved aspect via letterboxing on a white canvas and normalizes to mean/std of 0.5 per channel.

Example expected layout:

- ./data/captchas/ABC123-0.png → label “ABC123” with the default parsing logic.
- Configure IMG_HEIGHT/IMG_WIDTH and CHARS in config.py to match your dataset and vocabulary.

## Configuration

Key defaults are managed in Config within config.py and include image size, patch size, embedding dimension, depth, heads, dropout, max sequence length, training hyperparameters, and paths.

- Model: IMG_HEIGHT=64, IMG_WIDTH=256, PATCH_SIZE=8, EMBED_DIM=256, DEPTH=4, NUM_HEADS=4, MLP_RATIO=4.0, DROPOUT=0.2, MAX_SEQ_LEN=15, IN_CHANNELS=3.
- Training: BATCH_SIZE=32, NUM_EPOCHS=150, LEARNING_RATE=5e-4, LABEL_SMOOTHING=0.1, GRAD_CLIP_NORM=1.0, GRAD_ACCUM_STEPS=1, WEIGHT_DECAY=0.01, WARMUP_EPOCHS=0.
- Paths and device: DATA_DIR=./data/captchas, CHECKPOINT_DIR=./checkpoints, LOG_DIR=./logs, DEVICE auto-detected.

## Tokenizer

- CAPTCHATokenizer uses a configurable character string, builds char_to_idx and idx_to_char mappings, and appends a [PAD] token at the end with index equal to len(chars).
- encode(text, max_len=15) returns fixed-length sequences padded with PAD_IDX; decode and batch_decode remove padding to return strings.

## Model

- ViTSTR consists of a Conv2d patch embedder that maps (B,C,H,W) to (B,N,embed_dim) with kernel and stride equal to patch_size for non-overlapping patches.
- Learnable positional embeddings of shape (1, n_patches, embed_dim) are added, followed by dropout, a stack of Transformer encoder blocks, final LayerNorm, and a linear head applied per position to produce logits of shape (B, max_seq_len, num_classes).
- get_num_params and get_num_trainable_params are provided for introspection in training logs.

## Training

- Run training:

```bash
python train.py
```

This script reads config, builds tokenizer and dataloaders from DATA_DIR, constructs the ViTSTR model, and trains with CrossEntropyLoss using ignore_index=PAD_IDX and optional label_smoothing from config.

- train_epoch supports AMP on CUDA, gradient accumulation, gradient clipping, and reports running character accuracy; the script also logs validation loss, char accuracy, and sequence accuracy each epoch.
- Checkpoints are saved every epoch with metrics and optimizer state; the best model by val_seq_acc is exported to CHECKPOINT_DIR/best_model.pth.

## Evaluation

- The evaluate function computes average loss, character-level accuracy, and sequence-level accuracy while masking padding; it shows a tqdm progress bar with running metrics.
- evaluate_with_examples prints a few decoded true/pred pairs to help visually verify learning quality on the validation set.

## Inference

- After training, run single-image inference:

```bash
python inference.py path/to/image.png
```

The script builds tokenizer and model from config, loads CHECKPOINT_DIR/best_model.pth, preprocesses the image exactly as in training, and prints the predicted text.

## Checkpoints

- Per-epoch checkpoints store epoch number, model/optimizer state, and metrics via save_checkpoint; load_checkpoint restores model and optimizer states and returns the recorded epoch and metrics.
- save_best_model writes a compact file with model_state_dict and associated metrics when a new best val_seq_acc is achieved.

## Notes and customization

- If your repository is flat rather than packaged, change imports in train.py/inference.py from models/data/utils to direct module imports, or create packages exposing ViTSTR, CAPTCHADataset/CAPTCHATokenizer, and utility functions via **init**.py.
- Update CHARS and MAX_SEQ_LEN in Config to match the label space and maximum CAPTCHA length of your dataset, then retrain to reflect the new vocabulary and sequence length.

## Requirements

- Install exact or newer versions listed in requirements.txt before running training or inference to ensure API compatibility.
- PyTorch 2.0+ and TorchVision 0.15+ are specified, along with numpy, pillow, scipy, matplotlib, tqdm, and tensorboard versions suitable for training and logging.

## License

Add your preferred license file at the repository root; code here is provided without a license file in the current set of attachments, so choose and include one to clarify terms.
