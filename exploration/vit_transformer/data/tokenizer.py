"""
Character tokenizer for CAPTCHA text
"""

class CAPTCHATokenizer:
    """
    Tokenizer for encoding CAPTCHA text strings as integer indices, and decoding them back.
    
    Attributes:
        chars (str): Allowed character set.
        char_to_idx (dict): Character to integer index mapping.
        idx_to_char (dict): Integer index to character mapping.
        PAD_IDX (int): Index used for padding.
        vocab_size (int): Number of unique tokens (including padding).
    """
    
    def __init__(self, chars=None):
        """
        Initializes the tokenizer, builds dictionaries, and handles special tokens.

        Args:
            chars (str, optional): String of allowed characters. If None, uses default.
        """
        if chars is None:
            self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz'
        else:
            self.chars = chars
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        # Special tokens
        self.PAD_IDX = len(self.chars)
        self.char_to_idx['[PAD]'] = self.PAD_IDX
        self.idx_to_char[self.PAD_IDX] = '[PAD]'
        
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text, max_len=15):
        """
        Encodes a string label to a fixed-length list of token indices, with padding.

        Args:
            text (str): String label to encode.
            max_len (int): Fixed length for the output sequence.

        Returns:
            list: List of token indices (with padding if needed).
        """
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
        
        # Pad to max_len
        indices += [self.PAD_IDX] * (max_len - len(indices))
        return indices[:max_len]
    
    def decode(self, indices):
        """
        Decodes a list/tensor of indices back into a string.

        Args:
            indices (list or torch.Tensor): List of token indices.

        Returns:
            str: Decoded string (characters only, no padding).
        """
        chars = []
        for idx in indices:
            idx = int(idx)
            if idx != self.PAD_IDX and idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)
    
    def batch_decode(self, indices_batch):
        """
        Decodes a batch of encoded indices into strings.

        Args:
            indices_batch (list): List of list/tensor of indices.

        Returns:
            list: List of decoded strings.
        """
        return [self.decode(indices) for indices in indices_batch]
