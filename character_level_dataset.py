import torch
import numpy as np
from torch.utils.data import Dataset

class CharacterLevelDataset(Dataset):
    def __init__(self, data: np.ndarray, labels, max_seq_length, char_to_idx):
        self.data = data
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.char_to_idx = char_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Truncate or pad sequences to max_seq_length
        sequence = self.data[idx][:self.max_seq_length].ljust(self.max_seq_length)
        # Convert sequence to tensor of character indices
        sequence_tensor = torch.tensor([self.char_to_idx [char] for char in sequence])
        # Convert label to tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence_tensor, label_tensor, self.data[idx]