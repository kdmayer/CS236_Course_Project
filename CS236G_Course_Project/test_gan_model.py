import os
import pprint
import torch
import numpy as np
from src.model import Generator
from src.trainer import Trainer
from src.dataset import Lidar

root_dir = "."
data_dir = os.path.join(root_dir, "mock_data")
ckpt_path = os.path.join(root_dir, "checkpoints", "try_1", "0.pth")
seed = 0
split="Test"
# Number of points sampled from each training sample.
sample_size = 1000
batch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    """
    Testing entry point.
    """

    # Fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup dataloaders
    test_loader = torch.utils.data.DataLoader(
        dataset=Lidar(
            data_dir=data_dir,
            split="Test",
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # Setup model
    net_g = Generator()

    # Setup trainer
    trainer = Trainer(net_g=net_g, batch_size=batch_size, device=device)

    # Load checkpoint
    trainer.load_checkpoint(ckpt_path)

    # Start testing
    (metrics, submission), _ = trainer.test(test_loader)
    torch.set_printoptions(precision=6)
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()