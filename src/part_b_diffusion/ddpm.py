# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        
        noise_e = torch.randn_like(x)
        t = torch.randint(0, self.T, (x.shape[0], 1), device=x.device)
        noisy_image = torch.sqrt(self.alpha_cumprod[t]).reshape(-1, 1) * x  + torch.sqrt(1 - self.alpha_cumprod[t]).reshape(-1, 1) * noise_e
        t_norm = t.float() / self.T 
        equation = self.network(noisy_image, t_norm)
        neg_elbo = ((noise_e - equation)**2).mean(dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            if t > 1:
                z = torch.randn(shape).to(self.alpha.device)
            else:
                z = 0
            
            t_norm  = torch.full((shape[0], 1), t / self.T, device=x_t.device, dtype=torch.float)
            e_theta = self.network(x_t, t_norm)
            
            # Ensure e_theta doesn't have NaN
            if torch.isnan(e_theta).any():
                print(f"[WARNING] NaN in network output at step {t}, replacing with zeros")
                e_theta = torch.where(torch.isnan(e_theta), torch.zeros_like(e_theta), e_theta)
            
            at = self.alpha_cumprod[t].reshape(-1, 1)
            
            # Clamp at to avoid division by zero / extremely small values
            at = torch.clamp(at, min=1e-6)
            
            sqrt_at = torch.sqrt(at)
            sqrt_1_minus_at = torch.sqrt(torch.clamp(1 - at, min=1e-6))
            
            denoise = 1 / sqrt_at * (x_t - ((1 - at) / sqrt_1_minus_at * e_theta))
            
            # Clamp denoise to prevent extreme values
            denoise = torch.clamp(denoise, -10.0, 10.0)
            
            x_t = denoise + torch.sqrt(torch.clamp(self.beta[t], min=1e-8)) * z
            
            # Ensure x_t doesn't blow up
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                print(f"[WARNING] NaN/Inf in x_t at step {t}, replacing with zero")
                x_t = torch.where(torch.isnan(x_t) | torch.isinf(x_t), torch.zeros_like(x_t), x_t)
                            
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)