import torch
import torch.nn as nn
import torch.distributions as td


class Encoder(nn.Module):
    """
    Define a Gaussian encoder distribution based on a given encoder network.

    Parameters:
    encoder_net: [torch.nn.Module]             
       The encoder network that takes as a tensor of dim `(batch_size,
       feature_dim1, feature_dim2)` and output a tensor of dimension
       `(batch_size, 2M)`, where M is the dimension of the latent space.
    """
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        log_std = torch.clamp(log_std, -4, 4)   # numerical stability
        return td.Independent(td.Normal(mean, torch.exp(log_std)), 1)
