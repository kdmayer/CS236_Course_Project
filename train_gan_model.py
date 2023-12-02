import torch.utils.data
import click

from src.dataset import Lidar
from src.model import *
from src.trainer import Trainer
from src.metrics import *
from src.utils import *

from datetime import datetime


@click.command()
@click.option('--generator_loss', '-l', nargs=-1, type=str, default=None, help='Can contain the full set or any subset of "sinkhorn", "energy", "gaussian", "laplacian". If None, only the Chamfer Distance will be used.')
@click.option('--input_dir', '-i', type=str, default="mock_data", help="Must be either mock_data or data")
def main(generator_loss, input_dir):

    print(f"Using {input_dir} as input directory")
    print(f"Using {generator_loss} as generator loss")
    # Configuration
    root_dir = "./"
    data_dir = os.path.join(f"./{input_dir}")
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    # Name of current experiment. Checkpoints will be stored in '{ckpt_dir}/{name}/'.
    time_of_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name_of_run = str(generator_loss) + str(time_of_run)
    # Manual seed for reproducibility.
    seed = 0
    # Resumes training using the last checkpoint in ckpt_dir.
    resume = False
    batch_size = 32
    # Number of points sampled from each training sample.
    tr_sample_size = 500
    # Number of points sampled from each testing sample.
    te_sample_size = 500
    # Total training epoch.
    max_epoch = 2000
    # Number of discriminator updates before a generator update.
    repeat_d = 5
    log_every_n_step = 20
    val_every_n_epoch = 20
    ckpt_every_n_epoch = 100
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup checkpoint directory
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    ckpt_subdir = os.path.join(ckpt_dir, name_of_run)
    if not os.path.exists(ckpt_subdir):
        os.mkdir(ckpt_subdir)

    # Setup logging
    wandb.init(project="pcgan")

    # Setup dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=Lidar(
            data_dir=data_dir,
            split="Train",
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=Lidar(
            data_dir=data_dir,
            split="Val",
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # Setup model, optimizer and scheduler
    net_g = Generator()
    net_d = Discriminator()
    opt_g = torch.optim.Adam(net_g.parameters(), lr=4e-4, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=2e-4, betas=(0.9, 0.999))
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lambda e: 1.0)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lambda e: 1.0)

    # Setup trainer
    trainer = Trainer(
        net_g=net_g,
        net_d=net_d,
        opt_g=opt_g,
        opt_d=opt_d,
        sch_g=sch_g,
        sch_d=sch_d,
        device=device,
        batch_size=batch_size,
        max_epoch=max_epoch,
        repeat_d=repeat_d,
        log_every_n_step=log_every_n_step,
        val_every_n_epoch=val_every_n_epoch,
        ckpt_every_n_epoch=ckpt_every_n_epoch,
        ckpt_dir=ckpt_subdir,
        generator_loss_type=generator_loss
    )

    # Load checkpoint
    if resume:
        trainer.load_checkpoint()

    # Start training
    trainer.train(train_loader, val_loader)
    # Output displays: Loss of Generator, Loss of Discriminator, Train Epoch

if __name__ == '__main__':
    main()

