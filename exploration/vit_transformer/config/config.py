"""
Configuration file for ViTSTR CAPTCHA Recognition
"""
import torch

class Config:
    # Model Architecture
    IMG_HEIGHT = 64
    IMG_WIDTH = 256
    PATCH_SIZE = 8
    IN_CHANNELS = 3
    EMBED_DIM = 256
    DEPTH = 4
    NUM_HEADS = 4
    MLP_RATIO = 4.0
    DROPOUT = 0.2
    MAX_SEQ_LEN = 15
    
    # Character vocabulary
    CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz'
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    LEARNING_RATE = 5e-4
    MIN_LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 0
    GRAD_CLIP_NORM = 1.0
    GRAD_ACCUM_STEPS = 1
    LABEL_SMOOTHING = 0.1
    
    # Data
    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05
    NUM_WORKERS = 0
    PIN_MEMORY = False  

    # Paths
    DATA_DIR = './data/captchas'
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

