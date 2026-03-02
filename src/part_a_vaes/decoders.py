import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np


class BernoulliDecoder(nn.Module):
    """
    Define a Bernoulli decoder distribution based on a given decoder network.

    Parameters: 
    decoder_net: [torch.nn.Module]             
       The decoder network that takes as a tensor of dim `(batch_size, M) as
       input, where M is the dimension of the latent space, and outputs a
       tensor of dimension (batch_size, feature_dim1, feature_dim2).
    """
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)  # logits numerically more stable, Bernoulli will internally apply sigmoid
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GaussianDecoder(nn.Module):
    """
    Gaussian decoder: p(x|z) = N(x | μ(z), σ²I)
    
    Parameters:
    decoder_net: [torch.nn.Module]
        Network that outputs mean μ(z).
    std: [float]
        Standard deviation (default: 0.5).
    learnable_std: [bool]
        Whether std is learnable (default: False).
    """
    def __init__(self, decoder_net, std=0.5, learnable_std=False):
        super().__init__()
        self.decoder_net = decoder_net
        
        if learnable_std:
            self.log_std = nn.Parameter(torch.tensor(np.log(std)))
        else:
            self.register_buffer('log_std', torch.tensor(np.log(std)))
    
    def forward(self, z):
        """
        Given latent variables, return Gaussian distribution over data space.
        """
        mean = self.decoder_net(z)
        std = torch.exp(self.log_std)
        return td.Independent(td.Normal(mean, std), 2)
