import torch
import torch.utils.data
import numpy as np
import os

# The variable Split is either "Train", "Val", or "Test"
class Lidar(torch.utils.data.Dataset):
    def __init__(self, data_dir, split):
        self.data = []
        for fname in os.listdir(os.path.join(data_dir, split)):
            if fname.endswith(".npy"):
                path = os.path.join(data_dir, split, fname)
                # You add an extra dimension
                # Same as torch.unsqueeze but for numpy
                sample = np.load(path)[np.newaxis, ...]
                self.data.append(torch.from_numpy(sample).float())

        # Concat observations along first dim
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x