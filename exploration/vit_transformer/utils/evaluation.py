"""
Evaluation utilities - Fixed version
"""

import torch
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device, tokenizer):
    """
    Evaluate the model on a validation or test set.

    Args:
        model: Model to evaluate (in eval mode).
        dataloader: DataLoader for evaluation data.
        criterion: Loss function.
        device: torch.device for computation.
        tokenizer: Tokenizer to decode predictions.

    Returns:
        tuple:
            - avg_loss (float): Average loss over set.
            - char_accuracy (float): Character-level accuracy (0-1).
            - sequence_accuracy (float): Sequence-level accuracy (0-1).
    """
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_sequences = 0
    total_sequences = 0
    
    # Check if dataloader is empty
    if len(dataloader) == 0:
        print("WARNING: Empty validation dataloader")
        return 0.0, 0.0, 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            
            # Calculate loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            total_loss += loss.item()
            
            # Predictions
            predictions = logits.argmax(dim=-1)  # (B, max_seq_len)
            
            # Get padding index
            pad_idx = criterion.ignore_index
            
            # Character-level accuracy
            mask = labels != pad_idx
            correct_chars += ((predictions == labels) & mask).sum().item()
            total_chars += mask.sum().item()
            
            # Sequence-level accuracy
            # Check each sequence individually
            batch_size = labels.size(0)
            
            for i in range(batch_size):
                # Get non-padding positions for this sequence
                non_pad_mask = labels[i] != pad_idx
                num_chars = non_pad_mask.sum().item()
                
                # Only evaluate sequences with at least one character
                if num_chars > 0:
                    # Check if all non-padding characters are correct
                    correct_positions = (predictions[i] == labels[i]) & non_pad_mask
                    is_sequence_correct = correct_positions.sum().item() == num_chars
                    
                    if is_sequence_correct:
                        correct_sequences += 1
                    total_sequences += 1
            
            # Calculate running metrics
            char_acc = correct_chars / max(total_chars, 1)
            seq_acc = correct_sequences / max(total_sequences, 1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'char_acc': f'{char_acc:.4f}',
                'seq_acc': f'{seq_acc:.4f}'
            })
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    char_acc = correct_chars / max(total_chars, 1)
    seq_acc = correct_sequences / max(total_sequences, 1)
    
    return avg_loss, char_acc, seq_acc


def predict_batch(model, images, tokenizer, device):
    """
    Predict text for a batch of CAPTCHA images.

    Args:
        model: Trained model (in eval mode).
        images (torch.Tensor): Batch of images (B, C, H, W).
        tokenizer: CAPTCHATokenizer for decoding.
        device: torch.device for computation.

    Returns:
        list: List of predicted strings, one per image.
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=-1)  # (B, max_seq_len)
        
        # Decode predictions
        texts = tokenizer.batch_decode(predictions.cpu().numpy())
    
    return texts


def evaluate_with_examples(
    model, 
    dataloader, 
    criterion, 
    device, 
    tokenizer, 
    num_examples=5
):
    """
    Evaluate the model and print a few example predictions.

    Args:
        model: Model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: torch device.
        tokenizer: CAPTCHATokenizer for decoding.
        num_examples (int): Number of examples to print.

    Returns:
        tuple: (avg_loss (float), char_accuracy (float), sequence_accuracy (float))
    """
    avg_loss, char_acc, seq_acc = evaluate(model, dataloader, criterion, device, tokenizer)
    
    # Show some example predictions
    print("\n" + "="*80)
    print("Example Predictions:")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images[:num_examples].to(device)
        labels = labels[:num_examples].to(device)
        
        logits = model(images)
        predictions = logits.argmax(dim=-1)
        
        for i in range(min(num_examples, images.size(0))):
            true_text = tokenizer.decode(labels[i].cpu().numpy())
            pred_text = tokenizer.decode(predictions[i].cpu().numpy())
            match = "✓" if true_text == pred_text else "✗"
            print(f"{match} True: '{true_text}' | Pred: '{pred_text}'")
    
    print("="*80 + "\n")
    
    return avg_loss, char_acc, seq_acc
