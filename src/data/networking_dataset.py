import torch
import hashlib
import numpy as np
from torch.utils.data import Dataset

from data.transformations import jittering, positive_stretch


class NetworkingDataset(Dataset):
    def __init__(self, x, y, q, apply_transform=False, poison_ratio=0):
        super().__init__()
        self.x = x
        self.y = y
        self.q = q  # Quintuple of a biflow needed as random state to transform a biflow
        self.transform = None
        if apply_transform:
            self.transform = TwoViewsTrafficTransform(
                [jittering]
            )
        self.poisoning = None
        if poison_ratio > 0:
            self.poisoning = PoisoningTransform(poison_ratio)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        q = self.q[idx]

        if self.transform is None and self.poisoning is None:
            return torch.from_numpy(x).float(), torch.from_numpy(np.array(y)).float()

        if self.transform is not None:
            x_weak, x_strong = self.transform(x, random_state=q)

            x_weak = torch.from_numpy(x_weak).float()
            x_strong = torch.from_numpy(x_strong).float()
            y = torch.from_numpy(np.array(y)).float()

            return (x_weak, x_strong), y
        
        if self.poisoning is not None:
            if y.item() == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(np.array(y)).float()
            # Only poison the malicious samples (y=1)
            x_poisoned = self.poisoning(x, random_state=q)
            x_poisoned = torch.from_numpy(x_poisoned).float()
            y = torch.from_numpy(np.array(y)).float()

            return x_poisoned, y


class TwoViewsTrafficTransform:
    """
    TwoViewsTrafficTransform applies a random transformation from the provided list
    to generate two different views (weak and strong) of the same input biflow.
    """
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x, random_state):
        # Derive a seed integer from the random_state (quintuple)
        seed_int = int(hashlib.sha256(random_state.encode()).hexdigest(), 16) % (2**32)
        transformation = self.transformations[0]
        x_weak = transformation(x, a=0.5, random_state=seed_int)
        x_strong = transformation(x, a=2, random_state=seed_int + 1)
        return x_weak, x_strong
   
    
class PoisoningTransform:
    """
    PoisoningTransform adds a positive stretch to PL and IAT features of the input biflow, 
    simulating a specific type of adversarial attack.
    """
    def __init__(self, poison_ratio):
        self.poison_ratio = poison_ratio

    def __call__(self, x, random_state):
        # Derive a seed integer from the random_state (quintuple)
        seed_int = int(hashlib.sha256(random_state.encode()).hexdigest(), 16) % (2**32)
        return positive_stretch(x, a=self.poison_ratio, random_state=seed_int)
    