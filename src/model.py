import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np

class MaxBlock(nn.Module):
    """
    MaxBlock a modified linear layer
    """

    def __init__(self, in_dim, out_dim):
        """
        Parameters
        ----------
        in_dim: int
        Dimension of input
        out_dim: int
        Dimension of output
        Returns
        -------
        """
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # The input x is a torch.Tensor with shape [B x 1000 x 3], where B represents the batch size
        # xm is a torch.Tensor with shape [B x 1 x 3] and represents the maximum x, y, and z value for a point cloud with 1,000 points
        xm, _ = x.max(dim=1, keepdim=True)
        # We subtract the maximum values for the x, y, and z dimension from all the points in the point cloud
        x = self.proj(x - xm)
        return x

class Encoder(nn.Module):
    """
    Encoder takes in a point cloud and encodes it into a latent space.
    """

    def __init__(self, x_dim, d_dim, z1_dim):
        """
        Parameters
        ----------
        x_dim: int
        Dimension of input, here equal to three specifying a point's x,y, and z coordinate
        d_dim: int
        Dimension of hidden linear layers in the encoder MLP
        z1_dim: int
        Dimension of the latent space into which the point cloud is encoded
        Returns
        -------
        """
        super().__init__()
        self.phi = nn.Sequential(
            MaxBlock(x_dim, d_dim),
            nn.Tanh(),
            MaxBlock(d_dim, d_dim),
            nn.Tanh(),
            MaxBlock(d_dim, d_dim),
            nn.Tanh(),
        )
        self.ro = nn.Sequential(
            nn.Linear(d_dim, d_dim),
            nn.Tanh(),
            nn.Linear(d_dim, z1_dim),
        )

    def forward(self, x):
        x = self.phi(x)
        x, _ = x.max(dim=1)
        z1 = self.ro(x)
        return z1


class Decoder(nn.Module):
    """
    The Decoder creates a new point cloud by generating one 3D point at a time.
    """
    def __init__(self, x_dim, z1_dim, z2_dim, h_dim=512):
        """
        Parameters
        ----------
        x_dim: int
        Dimension of input, here equal to three specifying a point's x,y, and z coordinate
        z1_dim: int
        Dimension of the latent space into which the point cloud is encoded
        z2_dim: int
        Dimension of the noise vector which is added to the encoder latent space vector
        h_dim: int
        Dimension of the hidden layers in the decoder MLP
        Returns
        -------
        """

        super().__init__()
        # Linear layer maps latent space vector to h_dim
        self.fc = nn.Linear(z1_dim, h_dim)
        # Linear layer maps noise vector to h_dim
        self.fu = nn.Linear(z2_dim, h_dim, bias=False)
        self.dec = nn.Sequential(
            nn.Softplus(),
            # Check documentation of torch.nn.linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, x_dim),
        )

    def forward(self, z1, z2):
        """
        Parameters
        ----------
        z1: torch.Tensor
        Latent vector which represents encoded point cloud in the form [B x 1 x 256]
        z2: torch.Tensor
        Gaussian noise in the form [B x 1000 x 10]

        Returns
        -------
        o: torch.Tensor
        Generated output points represented as a tensor of shape [B x 1000 x 3]
        """

        # self.fc(z1) takes on the shape [B x 1 x 512]
        # self.fu(z2) takes on the shape [B x 1000 x 512]
        # x is the result of element-wise addition of the mapped latent vector and the mapped
        # noise vector, by default results in a tensor of shape [B x 1000 x 512]
        x = self.fc(z1) + self.fu(z2)
        o = self.dec(x)
        return o

    def latent_to_point_cloud(self, z1):
        """
        Parameters
        ----------
        z1: torch.Tensor
        Latent vector which represents encoded point cloud in the form [B x 1 x 256]

        Returns
        -------
        o: torch.Tensor
        Generated output points represented as a tensor of shape [B x 1000 x 3]
        """
        x = self.fc(z1)
        o = self.dec(x)
        return o

class Generator(nn.Module):
    """
    The Generator takes in a point cloud, encodes it into a latent space, adds random normal noise, and then creates
    a new point cloud by generating one 3D point at a time.
    """

    def __init__(self, x_dim=3, d_dim=256, z1_dim=256, z2_dim=10):
        """
        Parameters
        ----------
        x_dim: int
        Dimension of input, here equal to three specifying a point's x,y, and z coordinate
        d_dim: int
        Dimension of hidden linear layers in the encoder MLP
        z1_dim: int
        Dimension of the latent space into which the point cloud is encoded
        z2_dim: int
        Dimension of the noise vector which is added to the latent space vector
        Returns
        -------
        """
        super().__init__()
        self.z2_dim = z2_dim
        self.encoder_network = Encoder(x_dim, d_dim, z1_dim)
        self.decoder_network = Decoder(x_dim, z1_dim, z2_dim)

    def encode(self, real_point_cloud):
        """
        Parameters
        ----------
        real_point_cloud: torch.Tensor
        Input point cloud of shape [B x 1000 x 3]

        Returns
        -------
        z1: torch.Tensor
        Encoded point cloud of shape [B x 1 x 256]
        """
        # Calls the forward function of the encoder network
        z1 = self.encoder_network(real_point_cloud).unsqueeze(dim=1)
        return z1

    def decode(self, z1, B, N, device):
        """
        Parameters
        ----------
        z1: torch.Tensor
        Encoded point cloud of shape [B x 1 x 256]
        B: int
        batch size
        N: int, default is 1,000
        Number of points in the point cloud for a given object

        Returns
        -------
        generated_point_cloud: torch.Tensor
        Generated point cloud of dimension [B x 1000 x 3]
        """
        # z2 is the random noise which is later added to the encoded point cloud vector z1
        z2 = torch.randn((B, N, self.z2_dim)).to(device)
        # Calls the forward function of the decoder network
        # TODO: self.decoder_network() takes in the encoded point cloud vector z1 and the random noise vector z2.
        # TODO: Returns a generated point cloud of shape [B x 1000 x 3]
        # TODO: Think about how we can increase the number of generated points.
        generated_point_cloud = self.decoder_network(z1, z2)
        return generated_point_cloud

    def forward(self, real_point_cloud):
        """
        forward function of the Generator network

        Parameters
        ----------
        real_point_cloud: torch.Tensor
        Input point cloud of shape [B x 1000 x 3]

        Returns
        -------
        generated_point_cloud: torch.Tensor
        Generated point cloud of dimension [B x 1000 x 3]
        z1: torch.Tensor
        Encoded point cloud in the latent space with dimension [B x 1 x 256]
        """
        z1 = self.encode(real_point_cloud)
        # TODO: self.decode() takes in the encoded point cloud vector z1 and returns a generated point cloud of shape [B x 1000 x 3]
        # TODO: Think about how we can increase the number of generated points.
        generated_point_cloud = self.decode(z1, real_point_cloud.size(0), real_point_cloud.size(1), real_point_cloud.device)
        return generated_point_cloud, z1


class Discriminator(nn.Module):
    """
    The Discriminator takes in a point cloud and assigns a critic-like score with respect to how realistic it is
    """

    def __init__(self, x_dim=3, z1_dim=256, h_dim=1024, o_dim=1):
        """
        Parameters
        ----------
        x_dim: int, default 3
        Dimension of input, here equal to three specifying a point's x,y, and z coordinate
        z1_dim: int, default 256
        Dimension of the latent space into which the point cloud has been encoded
        h_dim: int, default 1024
        Dimension of the hidden linear layers in the discriminator MLP
        o_dim: int, default 1
        Critic score
        Returns
        -------
        """

        super().__init__()
        self.fc = nn.Linear(z1_dim, h_dim)
        self.fu = nn.Linear(x_dim, h_dim, bias=False)

        self.d1 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim - z1_dim),
        )

        self.sc = nn.Linear(z1_dim, h_dim)
        self.su = nn.Linear(h_dim - z1_dim, h_dim, bias=False)

        self.d2 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim - z1_dim),
        )

        self.tc = nn.Linear(z1_dim, h_dim)
        self.tu = nn.Linear(h_dim - z1_dim, h_dim, bias=False)
        self.d3 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, o_dim),
            # You can add a Softmax here and adjust the o_dim to be the number of classes
            # This one is currently conditioned on z1
        )

    def forward(self, real_point_cloud, z1):
        """
        forward function of the Discriminator network. Generator aims to maximize the critic score,
        while the Discriminator aims to minimize the critic score for generated examples.

        Parameters
        ----------
        real_point_cloud: torch.Tensor
        Real point cloud of shape [B x 1000 x 3]
        z1: torch.Tensor
        Encoded point cloud represented as a latent vector of shape [B x 1 x 256]

        Returns
        -------
        critic_score: torch.Tensor
        critic score assigned to mixture, i.e. sum, of the real point cloud x which is mapped to h_dim and the
        latent representation of the point cloud z1 which is mapped to h_dim
        """

        # Note 1: fc, sc, and tc are simply linear mappings (torch.nn.Linear) for the encoded point cloud
        # Note 2: fu is simply a linear mapping (torch.nn.Linear) for the original point cloud
        # Note 3: d1, d2, and d3 are shallow neural networks with linear layers and softplus activations
        y = self.fc(z1) + self.fu(real_point_cloud)
        o = self.d1(y)
        y = self.sc(z1) + self.su(o)
        o = self.d2(y)
        y = self.tc(z1) + self.tu(o)
        critic_score = self.d3(y)
        return critic_score