"""
Run inference on single or batch of CAPTCHA images
"""

import sys

import torch

from config import Config
from models import ViTSTR
from data import CAPTCHATokenizer, CAPTCHADataset

def predict_single_image(model, image_path, tokenizer, device, cfg):
    """Predict text from single image"""
    # Create temporary dataset for preprocessing
    dataset = CAPTCHADataset(
        [image_path],
        ['dummy'], 
        cfg.IMG_HEIGHT,
        cfg.IMG_WIDTH,
        tokenizer
    )
    
    img_tensor, _ = dataset[0]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        predictions = logits.argmax(dim=-1)
    
    # Decode
    text = tokenizer.decode(predictions[0].cpu().numpy())
    
    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    cfg = Config()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer
    tokenizer = CAPTCHATokenizer(chars=cfg.CHARS)
    
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
        dropout=0.0  # No dropout for inference
    ).to(device)
    
    # Load best checkpoint
    checkpoint_path = f'{cfg.CHECKPOINT_DIR}/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")
    
    # Predict
    predicted_text = predict_single_image(model, image_path, tokenizer, device, cfg)
    
    print("\n" + "="*80)
    print(f"Image: {image_path}")
    print(f"Predicted text: {predicted_text}")
    print("="*80)


if __name__ == '__main__':
    main()
