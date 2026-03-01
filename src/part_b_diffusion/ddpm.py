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
            at = self.alpha_cumprod[t].reshape(-1, 1)
            denoise = 1 / torch.sqrt(at) * (x_t - ((1 - at) / torch.sqrt(1 - at) * e_theta))
            x_t = denoise + torch.sqrt(self.beta[t]) * z
                            
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


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData
    from unet import Unet

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard, mnist} (default: %(default)s)')
    parser.add_argument('--network', type=str, default='fc', choices=['fc', 'unet'], help='network architecture to use {fc: fully connected, unet} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='ckpts/model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples/samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Define preprocessing for MNIST
    def preprocess_mnist(x):
        """Dequantize and scale MNIST images to [-1, 1]"""
        x = x.float() / 255.0
        x = x + torch.rand_like(x) / 256.0  # Add dequantization noise
        x = (x - 0.5) * 2.0  # Scale to [-1, 1]
        return x

    # Generate the data
    if args.data == 'mnist':
        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='data', train=True, download=False, 
                                       transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='data', train=False, download=False, 
                                      transform=transforms.ToTensor())
        
        # Preprocess and prepare data based on network type
        if args.network == 'fc':
            # Flatten images for FC network
            def collate_fn_fc(batch):
                images, _ = zip(*batch)
                images = torch.stack(images)
                images = preprocess_mnist(images).reshape(len(images), -1)
                return images
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                       shuffle=True, collate_fn=collate_fn_fc)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                                      shuffle=False, collate_fn=collate_fn_fc)
            D = 28 * 28
        else:  # unet
            # Keep images for U-Net
            def collate_fn_unet(batch):
                images, _ = zip(*batch)
                images = torch.stack(images)
                images = preprocess_mnist(images).reshape(len(images), -1)
                return images
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                       shuffle=True, collate_fn=collate_fn_unet)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                                      shuffle=False, collate_fn=collate_fn_unet)
            D = 28 * 28
    else:
        # Toy datasets
        n_data = 10000000
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        D = next(iter(train_loader)).shape[1]

    # Define the network
    num_hidden = 64
    if args.network == 'fc':
        network = FcNetwork(D, num_hidden)
    else:  # unet
        network = Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Ensure output directories exist
    import os
    os.makedirs('Week3/outputs', exist_ok=True)
    os.makedirs('Week3/ckpts', exist_ok=True)
    os.makedirs('Week3/samples', exist_ok=True)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

        # Save training loss to outputs
        with open('Week3/outputs/elbo.txt', 'a') as f:
            f.write(f'{args.model}: trained\n')

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000, D))).cpu() 

        if args.data == 'mnist':
            # For MNIST, transform back to [0, 1] and save as image grid
            samples = samples / 2 + 0.5  # Transform from [-1, 1] to [0, 1]
            samples = samples.reshape(10000, 1, 28, 28)
            
            # Save a grid of samples
            grid_size = 10
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    axes[i, j].imshow(samples[idx, 0].cpu().numpy(), cmap='gray')
                    axes[i, j].axis('off')
            plt.tight_layout()
            plt.savefig(args.samples, dpi=100)
            plt.close()
            
            # Save samples as numpy array for evaluation
            np.save(args.samples.replace('.png', '.npy'), samples.numpy())
        else:
            # For toy datasets, plot the density
            samples = samples / 2 + 0.5  # Transform back to [0, 1]
            
            # Plot the density of the toy data and the model samples
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()
