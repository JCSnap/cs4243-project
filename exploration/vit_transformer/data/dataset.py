"""
PyTorch Dataset for CAPTCHA images
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CAPTCHADataset(Dataset):
    """
    PyTorch Dataset for CAPTCHA images supporting variable image widths,
    and preprocessing suitable for ViTSTR models.

    Attributes:
        image_paths (List[str]): List of image file paths.
        labels (List[str]): Corresponding list of text labels for each image.
        img_height (int): Target image height.
        img_width (int): Target image width.
        tokenizer (CAPTCHATokenizer): Tokenizer for encoding labels.
        normalize (torchvision.transforms.Compose): Normalization pipeline.
    """
    
    def __init__(
        self,
        image_paths,
        labels,
        img_height=32,
        img_width=128, 
        tokenizer=None
    ):
        """
        Initialize the dataset, prepare normalization pipeline.

        Args:
            image_paths (list): List of image file paths.
            labels (list): List of string labels.
            img_height (int): Target image height after resize/pad.
            img_width (int): Target image width after resize/pad.
            tokenizer (CAPTCHATokenizer): Tokenizer instance for encoding labels.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width
        self.tokenizer = tokenizer
        
        # Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def resize_with_padding(self, img):
        """
        Resize an input image to target dimensions while maintaining aspect ratio,
        pads with white if needed.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Padded and resized image.
        """
        old_width, old_height = img.size
        
        # Calculate scaling factor
        scale = min(self.img_width / old_width, self.img_height / old_height)
        scale = min(scale, 1.0)
        new_width = int(old_width * scale)
        new_height = int(old_height * scale)
        
        # Resize
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with white padding
        new_img = Image.new('RGB', (self.img_width, self.img_height), (255, 255, 255))
        
        # Paste resized image centered
        paste_x = (self.img_width - new_width) // 2
        paste_y = (self.img_height - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    
    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of images/labels available.
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retrieve, process, and return one sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            img_tensor (torch.Tensor): Normalized image tensor of shape (C, H, W).
            label_tensor (torch.LongTensor): Encoded label tensor of length max_seq_len.
        """
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Resize with padding
        img = self.resize_with_padding(img)
        
        # Convert to tensor and normalize
        img_tensor = self.normalize(img)
        
        # Tokenize label
        label = self.labels[idx]
        label_indices = self.tokenizer.encode(label)
        label_tensor = torch.LongTensor(label_indices)
        
        return img_tensor, label_tensor


def create_data_loaders(
    train_paths,
    train_labels,
    val_paths, 
    val_labels,
    tokenizer, 
    batch_size=32,
    num_workers=0,
    img_height=64, 
    img_width=256
):
    """
    Construct PyTorch DataLoaders for both training and validation datasets.

    Args:
        train_paths (list): Training image file paths.
        train_labels (list): Training text labels.
        val_paths (list): Validation image file paths.
        val_labels (list): Validation text labels.
        tokenizer (CAPTCHATokenizer): Tokenizer for encoding.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of data loading workers.
        img_height (int): Image height after preprocessing.
        img_width (int): Image width after preprocessing.

    Returns:
        tuple: (train_loader, val_loader) as torch.utils.data.DataLoader
    """
    # Training dataset
    train_dataset = CAPTCHADataset(
        train_paths,
        train_labels,
        img_height,
        img_width, 
        tokenizer
    )
    
    # Validation dataset
    val_dataset = CAPTCHADataset(
        val_paths,
        val_labels,
        img_height,
        img_width, 
        tokenizer
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False  # Reuse workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader
