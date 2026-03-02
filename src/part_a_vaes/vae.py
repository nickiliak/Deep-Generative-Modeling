import torch
import torch.nn as nn
import torch.distributions as td

try:
    from .priors import GaussianPrior
except ImportError:
    from priors import GaussianPrior


class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder, beta=1.0):
        """
        Variational Autoencoder with optional β weighting for KL divergence.

        Parameters:
        prior: Prior distribution over latent space (Gaussian, MoG, or Flow)
        decoder: Decoder distribution over data space
        encoder: Encoder distribution over latent space
        beta: Weight for KL divergence term (default: 1.0). Use beta < 1 for more 
              reconstruction focus, beta > 1 for more disentanglement.
        """
        super().__init__()
        self.prior   = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x):
        """
        ELBO = E_q[log p(x|z)] - β*KL(q(z|x) || p(z))

        - GaussianPrior: KL in closed form (exact, most stable).
        - MoGPrior:      Monte Carlo KL: log q(z) - log p(z),  z ~ q(z|x)
        - FlowPrior:     Monte Carlo KL: log q(z) - log p_flow(z),  z ~ q(z|x)
                         where log p_flow uses the inverse flow + base density.
        
        The KL term is weighted by self.beta to support β-VAE variants.
        """
        q   = self.encoder(x)
        z   = q.rsample()
        rec = self.decoder(z).log_prob(x)

        if isinstance(self.prior, GaussianPrior):
            kl = td.kl_divergence(q, self.prior())        # closed-form
        else:
            # MC estimate — works for MoG and Flow (any prior with log_prob)
            log_q = q.log_prob(z)                         # log q(z|x)
            log_p = self.prior.log_prob(z)                # log p(z)
            kl    = log_q - log_p                         # shape: (batch,)

        return torch.mean(rec - self.beta * kl)

    def sample(self, n=1):
        """
        Sample n images: z ~ p(z), x ~ p(x|z).
        Returns hard binary Bernoulli samples.
        Calls prior.sample() directly so it works for all three prior types.
        """
        with torch.no_grad():
            if isinstance(self.prior, GaussianPrior):
                z = self.prior().sample(torch.Size([n]))
            else:
                z = self.prior.sample(n)                  # MoG and Flow
            return self.decoder(z).sample()               # (n, 28, 28)

    def forward(self, x):
        return -self.elbo(x)
