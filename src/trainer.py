import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
import os
from src.metrics import *
from src.utils import plot_samples

class Trainer:
    def __init__(
        self,
        net_g,
        device,
        batch_size,
        net_d=None,
        opt_g=None,
        opt_d=None,
        sch_g=None,
        sch_d=None,
        max_epoch=None,
        repeat_d=None,
        log_every_n_step=None,
        val_every_n_epoch=None,
        ckpt_every_n_epoch=None,
        ckpt_dir=None,
        generator_loss_type=None
    ):
        self.net_g = net_g.to(device)
        self.device = device
        self.batch_size = batch_size
        self.net_d = net_d and net_d.to(device)
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.sch_g = sch_g
        self.sch_d = sch_d
        self.step = 0
        self.epoch = 0
        self.max_epoch = max_epoch
        self.repeat_d = repeat_d
        self.log_every_n_step = log_every_n_step
        self.val_every_n_epoch = val_every_n_epoch
        self.ckpt_every_n_epoch = ckpt_every_n_epoch
        self.ckpt_dir = ckpt_dir
        self.generator_loss_type = generator_loss_type

    def _state_dict(self):
        return {
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sch_g": self.sch_g.state_dict(),
            "sch_d": self.sch_d.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "max_epoch": self.max_epoch,
        }

    def _load_state_dict(self, state_dict):
        for k, m in {
            "net_g": self.net_g,
            "net_d": self.net_d,
            "opt_g": self.opt_g,
            "opt_d": self.opt_d,
            "sch_g": self.sch_g,
            "sch_d": self.sch_d,
        }.items():
            m and m.load_state_dict(state_dict[k])
        self.step, self.epoch, self.max_epoch = map(
            state_dict.get,
            (
                "step",
                "epoch",
                "max_epoch",
            ),
        )

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"{self.epoch}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def load_checkpoint(self, ckpt_path=None):
        if not ckpt_path:  # Find last checkpoint in ckpt_dir
            ckpt_paths = [p for p in os.listdir(self.ckpt_dir) if p.endswith(".pth")]
            assert ckpt_paths, "No checkpoints found."
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
        self._load_state_dict(torch.load(ckpt_path))

    def _train_step_g(self, real_point_cloud):
        generated_point_cloud, z1 = self.net_g(real_point_cloud)
        discriminator_output = self.net_d(generated_point_cloud, z1.detach())
        loss_hinge = -discriminator_output.mean()
        loss_cd = compute_cd(generated_point_cloud, real_point_cloud, reduce_func=torch.sum).mean()

        # TODO: Modify the loss computation to take into account a combination of different losses?
        if ("cuda" in self.device) and (self.generator_loss_type in ["sinkhorn", "energy", "gaussian"]):
            import geomloss
            # Define loss between sampled measures
            loss = geomloss.SamplesLoss(loss="gaussian", p=2, blur=.05)
            geom_loss = loss(generated_point_cloud, real_point_cloud)
            mean_geom_loss = geom_loss.mean() # average loss_sinkhorn over all elements in the batch
            return loss_hinge + loss_cd + mean_geom_loss

        return loss_hinge + loss_cd

    def _train_step_d(self, real_point_cloud):
        generated_point_cloud, z1 = self.net_g(real_point_cloud)
        critic_score_real = self.net_d(real_point_cloud, z1.detach())
        critic_score_fake = self.net_d(generated_point_cloud.detach(), z1.detach())
        loss_discriminator = F.relu(1.0 - critic_score_real).mean() + F.relu(1.0 + critic_score_fake).mean()
        return loss_discriminator

    def train(self, train_loader, val_loader):
        while self.epoch < self.max_epoch:

            # Validation and checkpointing
            if self.epoch % self.val_every_n_epoch == 0:
                (metrics, _), samples = self.test(val_loader)
                wandb.log({**metrics, "samples": samples, "epoch": self.epoch})
            if self.epoch % self.ckpt_every_n_epoch == 0:
                self.save_checkpoint()

            with tqdm(train_loader) as t:
                self.net_g.train()
                self.net_d.train()
                for batch in t:

                    # Update step
                    loss_d = self._train_step_d(batch.to(self.device))
                    self.opt_d.zero_grad()
                    loss_d.backward()
                    self.opt_d.step()
                    if self.step % self.repeat_d == 0:
                        loss_g = self._train_step_g(batch.to(self.device))
                        self.opt_g.zero_grad()
                        loss_g.backward()
                        self.opt_g.step()

                    # Stepwise logging
                    t.set_description(
                        f"Epoch:{self.epoch}|L(G):{loss_g.item():.2f}|L(D):{loss_d.item():.2f}"
                    )
                    if self.step % self.log_every_n_step == 0:
                        wandb.log(
                            {
                                "loss_g": loss_g.cpu(),
                                "loss_d": loss_d.cpu(),
                                "step": self.step,
                                "epoch": self.epoch,
                            }
                        )

                    self.step += 1
                self.sch_g.step()
                self.sch_d.step()
            self.epoch += 1

    def _test_step(self, real_point_cloud):
        generated_point_cloud, _ = self.net_g(real_point_cloud)
        return generated_point_cloud, real_point_cloud

    def _test_end(self, generated_point_cloud, real_point_cloud):
        # generated_point_cloud, [B x Number of Points x 3]
        # real_point_cloud, [B x Number of Points x 3]
        metrics = compute_metrics(generated_point_cloud, real_point_cloud, self.batch_size)
        samples = plot_samples(generated_point_cloud)
        return metrics, samples

    @torch.no_grad()
    def test(self, test_loader):
        results = []
        self.net_g.eval()
        for batch in tqdm(test_loader):
            results.append(self._test_step(batch.to(self.device)))
        return self._test_end(*(torch.cat(_, dim=0) for _ in zip(*results)))
