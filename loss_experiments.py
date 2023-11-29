from geomloss import SamplesLoss
import numpy as np
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load example point cloud 1
pc_1 = torch.from_numpy(np.load("./mock_data/Train/-1.490470464058936,52.41507789289978.npy")).to(device)

# Load example point cloud 2
pc_2 = torch.from_numpy(np.load("./mock_data/Train/-1.490525777723789,52.416252726220506.npy")).to(device)

def compute_sinkhorn_loss():

    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    L = loss(pc_1, pc_2)  # By default, use constant weights = 1/number of samples

    print("***")
    print("Computed Loss:", L)
    print("***")

if __name__ == "__main__":
    compute_sinkhorn_loss()