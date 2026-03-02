import torch
import torch.nn as nn
import torch.distributions as td


class GaussianPrior(nn.Module):
    """Standard isotropic Gaussian prior: p(z) = N(0, I)."""

    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std  = nn.Parameter(torch.ones(M),  requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)


class MoGPrior(nn.Module):
    """
    Mixture of Gaussians prior:
        p(z) = sum_k w_k * N(z | mu_k, sigma_k^2 * I)

    The means, stds, and weights are fixed (not learned) to keep the prior
    stable during training. The means are spread via randn so components
    are initialised at different locations in latent space.

    KL(q || p) has no closed form -> computed via Monte Carlo in VAE.elbo().
    """

    def __init__(self, M, K=10):
        super().__init__()
        self.M = M
        self.K = K
        self.means   = nn.Parameter(torch.randn(K, M),  requires_grad=False)
        self.stds    = nn.Parameter(torch.ones(K, M),   requires_grad=False)
        self.weights = nn.Parameter(torch.ones(K) / K,  requires_grad=False)

    def log_prob(self, z):
        """Compute the log-probability of z under the mixture distribution."""
        return self.forward().log_prob(z)

    def sample(self, n):
        """Sample n points from the mixture distribution."""
        return self.forward().sample(torch.Size([n]))

    def forward(self):
        mix  = td.Categorical(probs=self.weights)
        comp = td.Independent(td.Normal(self.means, self.stds), 1)
        return td.MixtureSameFamily(mix, comp)


class MaskedCouplingLayer(nn.Module):
    """
    Affine coupling layer (RealNVP).

    Given input z and a binary mask m:
      - Masked half:     z1 = z * m          (passed through unchanged)
      - Transformed half: z2' = z*(1-m) * exp(s(z1)) + t(z1)

    The log-determinant of the Jacobian is sum(s * (1-m)) — only the
    unmasked dimensions are scaled, so the Jacobian is triangular.

    Inverse:
      z2 = (z2' - t(z1)) * exp(-s(z1))
    """

    def __init__(self, scale_net, translation_net, mask):
        super().__init__()
        self.scale_net       = scale_net
        self.translation_net = translation_net
        self.mask            = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """Forward pass: z -> z'. Returns (z_out, log_det_J)."""
        z1  = z * self.mask                          # masked (unchanged) part
        s   = self.scale_net(z1)
        t   = self.translation_net(z1)
        z_out     = z1 + (z * (1 - self.mask)) * torch.exp(s) + t * (1 - self.mask)
        log_det_J = (s * (1 - self.mask)).sum(dim=1)
        return z_out, log_det_J

    def inverse(self, z_out):
        """Inverse pass: z' -> z. Returns (z, log_det_J)."""
        z1      = z_out * self.mask
        s       = self.scale_net(z1)
        t       = self.translation_net(z1)
        z2      = (z_out * (1 - self.mask) - t * (1 - self.mask)) * torch.exp(-s)
        z       = z1 + z2
        log_det_J = -(s * (1 - self.mask)).sum(dim=1)
        return z, log_det_J


class FlowPrior(nn.Module):
    """
    Normalising flow prior built from stacked affine coupling layers.

    The generative direction is:  z0 ~ N(0,I)  ->  z = f(z0)
    So to evaluate log p(z) we run the INVERSE:
        z0, log|det J_f^-1| = f^-1(z)
        log p(z) = log N(z0; 0,I) + log|det J_f^-1|

    Sampling:
        z0 ~ N(0, I)  ->  z = f(z0)  (forward pass through all layers)

    The flow parameters ARE learned jointly with the encoder/decoder.
    """

    def __init__(self, M, num_layers=4, hidden_dim=64):
        super().__init__()
        self.M    = M
        # Fixed base distribution N(0, I)
        self.base_mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.base_std  = nn.Parameter(torch.ones(M),  requires_grad=False)

        # Stack of alternating coupling layers
        self.layers = nn.ModuleList()
        # Start mask: first half ones, second half zeros
        mask = torch.cat([torch.ones(M // 2), torch.zeros(M - M // 2)])
        for _ in range(num_layers):
            mask = 1 - mask    # alternate what half is masked each layer
            scale_net = nn.Sequential(
                nn.Linear(M, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, M), nn.Tanh(),   # Tanh keeps scales bounded
            )
            translation_net = nn.Sequential(
                nn.Linear(M, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, M),
            )
            self.layers.append(MaskedCouplingLayer(scale_net, translation_net, mask.clone()))

    def base(self):
        return td.Independent(td.Normal(self.base_mean, self.base_std), 1)

    def forward(self, z):
        """Run z through inverse flow to get z0 and accumulated log|det J|."""
        log_det_J = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):   # inverse = reverse order
            z, ldj = layer.inverse(z)
            log_det_J += ldj
        return z, log_det_J

    def log_prob(self, z):
        """
        Compute log p(z) under the flow prior.
        log p(z) = log p_base(f^-1(z)) + log|det J_f^-1(z)|
        """
        z0, log_det_J = self.forward(z)
        return self.base().log_prob(z0) + log_det_J

    def sample(self, n):
        """
        Sample from the flow prior.
        z0 ~ N(0,I), then z = f(z0) via forward pass through all layers.
        """
        z = self.base().sample(torch.Size([n]))
        log_det_J = torch.zeros(n, device=z.device)
        for layer in self.layers:             # generative = forward order
            z, ldj = layer.forward(z)
            log_det_J += ldj
        return z
