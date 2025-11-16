"""
Data Preprocessing Script for CRNN Captcha Model

This script performs the following steps:
1. Finds all .png images in the SOURCE_DIR.
2. Validates and filters image paths based on their filenames (labels).
3. Loads, decodes, resizes, and normalizes each valid image.
4. Encodes the string labels into padded integer arrays.
5. Saves the processed images and labels into a single compressed .npz file.

This file is intended to be run ONCE before training.
The 'crnn.py' script will then load the output of this file.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

# --- Constants ---

SOURCE_DIR = "/path/to/your/raw/images" 

# Image dimensions (must match crnn.py)
IMG_WIDTH = 550
IMG_HEIGHT = 80

# Output file (must match crnn.py)
SAVE_PREPROCESSED_IMAGES_DIR = 'crnn_preprocessed_images.npz'

# Alphabet (must match crnn.py)
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
char_to_idx = {c: i for i, c in enumerate(ALPHABET)}


def validate_and_filter_paths(image_folder_path):
    """
    Scans the source directory, validates labels, and returns clean lists.
    """
    print("Scanning and validating data...")
    image_folder = Path(image_folder_path)
    all_image_paths = sorted(list(image_folder.glob("*.png")))
    allowed_chars = set(ALPHABET)

    print(f"Found {len(all_image_paths)} total images.")

    valid_image_paths = []
    valid_labels = []

    for path in tqdm(all_image_paths, desc="Validating files"):
        # get label from filename
        label = path.name.split('-')[0].lower()

        # keep only allowed chars
        filtered = "".join([c for c in label if c in allowed_chars])

        if len(filtered) == 0:
            # print(f"[Skip] {path.name} -> label '{label}' invalid after filtering.")
            continue

        if filtered != label:
            print(f"[Clean] {path.name} -> '{label}' -> '{filtered}'")
        
        # Check if all characters are in our alphabet
        if not all(c in char_to_idx for c in filtered):
            # print(f"[Skip/Error] {path.name} ('{filtered}') contains unknown chars.")
            continue

        valid_image_paths.append(str(path)) # Store as string
        valid_labels.append(filtered)

    print(f"Using {len(valid_image_paths)} / {len(all_image_paths)} valid images.")
    return valid_image_paths, valid_labels


def process_single_image(image_path):
    """
    Loads, decodes, resizes, and normalizes a single image.
    """
    try:
        img = tf.io.read_file(image_path)
        img = tf.io.decode_png(img, channels=1) # Force grayscale
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.transpose(img, perm=[1, 0, 2]) # (H, W, 1) -> (W, H, 1) if needed, check model
        
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        # The original model's input is (H, W, 1), so let's transpose back
        # The Conv2D expects (batch, H, W, C)
        # Your model: layers.Input((img_height, img_width, 1))
        # So we should NOT transpose.
        
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def encode_and_pad_labels(labels, max_len=None):
    """
    Encodes string labels to int arrays and pads with -1.
    """
    encoded_labels = []
    
    # 1. Encode all labels
    for label in labels:
        indices = [char_to_idx[ch] for ch in label]
        encoded_labels.append(indices)
        
    # 2. Find max length if not provided
    if max_len is None:
        max_len = max(len(lab) for lab in encoded_labels)
    print(f"Max label length found: {max_len}")
    
    # 3. Pad with -1 (as expected by CTCLayer)
    padded_labels = []
    for lab in encoded_labels:
        padding = [-1] * (max_len - len(lab))
        padded_lab = lab + padding
        padded_labels.append(padded_lab)
        
    return np.array(padded_labels, dtype="int32")


def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: SOURCE_DIR not found at '{SOURCE_DIR}'")
        print("Please update the SOURCE_DIR variable in this script.")
        return

    # 1. Get filtered file lists
    valid_paths, valid_str_labels = validate_and_filter_paths(SOURCE_DIR)
    
    if not valid_paths:
        print("No valid image paths found. Exiting.")
        return

    # 2. Process images
    images_list = []
    labels_to_keep = []
    
    for path, label in tqdm(zip(valid_paths, valid_str_labels), 
                            total=len(valid_paths), 
                            desc="Processing images"):
        img = process_single_image(path)
        if img is not None:
            images_list.append(img.numpy()) # Convert tensor to numpy
            labels_to_keep.append(label)

    # Convert images list to a single numpy array
    images_np = np.array(images_list, dtype="float32")
    print(f"Processed images shape: {images_np.shape}")

    # 3. Encode and pad labels
    labels_np = encode_and_pad_labels(labels_to_keep)
    print(f"Processed labels shape: {labels_np.shape}")

    # 4. Save to .npz file
    print(f"Saving data to {SAVE_PREPROCESSED_IMAGES_DIR}...")
    np.savez_compressed(
        SAVE_PREPROCESSED_IMAGES_DIR,
        images=images_np,
        labels=labels_np
    )
    print("Data saved successfully.")


if __name__ == "__main__":
    main()