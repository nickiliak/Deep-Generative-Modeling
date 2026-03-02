import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm



# Priors

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



# Flow-based Prior

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


#define the enc. / dec. and training loop based on exercises week 1

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


class BernoulliDecoder(nn.Module):
    """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
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
        logits = self.decoder_net(z) #logits numerically more stable, Benroulli will internally apply sigmoid
        return td.Independent(td.Bernoulli(logits=logits), 2)


# VAE

class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior   = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

        - GaussianPrior: KL in closed form (exact, most stable).
        - MoGPrior:      Monte Carlo KL: log q(z) - log p(z),  z ~ q(z|x)
        - FlowPrior:     Monte Carlo KL: log q(z) - log p_flow(z),  z ~ q(z|x)
                         where log p_flow uses the inverse flow + base density.
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

        return torch.mean(rec - kl)

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


# Training

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    total_steps = len(data_loader) * epochs
    bar = tqdm(range(total_steps), desc="Training")
    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
            bar.update()


# Evaluation: Test-set ELBO

def evaluate_elbo(model, data_loader, device):
    """Return mean ELBO over the full test loader."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in data_loader:
            total += model.elbo(x.to(device)).item()
            n     += 1
    return total / n


# This is for 
# Multi-run evaluation (mean +/- std)

def multirun_evaluation(prior_name, M, train_loader, test_loader, epochs, lr, device, n_runs,
                        flow_layers=4, flow_hidden=64):
    """
    Trains the model from scratch n_runs times and reports mean +/- std of
    the test-set ELBO across runs
    """
    elbos = []
    for run in range(1, n_runs + 1):
        print(f"\n-- Run {run}/{n_runs} --")
        model     = build_model(prior_name, M, device, flow_layers, flow_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train(model, optimizer, train_loader, epochs, device)
        elbo = evaluate_elbo(model, test_loader, device)
        elbos.append(elbo)
        print(f"  Run {run} test ELBO: {elbo:.4f}")

    elbos      = np.array(elbos)
    mean, std  = elbos.mean(), elbos.std()
    label      = PRIOR_LABELS[prior_name]

    print(f"\n{'─'*50}")
    print(f"[Multi-run results -- {label} prior]")
    print(f"  Runs:    {n_runs}")
    print(f"  ELBOs:   {np.round(elbos, 4)}")
    print(f"  Mean:    {mean:.4f}")
    print(f"  Std:     {std:.4f}")
    print(f"  Summary: {mean:.4f} +/- {std:.4f}")
    print(f"{'─'*50}")
    return mean, std, elbos


# Plotting: prior vs aggregate posterior

PRIOR_LABELS = {
    "gaussian": "Gaussian",
    "mog":      "Mixture of Gaussians",
    "flow":     "Flow-based",
}


def _sample_prior(model, N):
    """Helper: sample N points from any prior type."""
    with torch.no_grad():
        if isinstance(model.prior, GaussianPrior):
            return model.prior().sample(torch.Size([N])).cpu().numpy()
        else:
            return model.prior.sample(N).cpu().numpy()


def plot_prior_vs_posterior(model, data_loader, device, prior_name, save_path, n_batches=50):
    """
    Two-panel plot comparing prior and aggregate posterior.

    Following the project spec exactly:
      - Posterior points are SAMPLES from q(z|x)  [not the mean]
      - If M > 2: PCA projects samples onto the first 2 principal components
      - If M == 2: use the dimensions directly
      - Prior is also projected through the same PCA fit (if M > 2)
      - Points coloured by digit class (tab10)
      - KDE shown as smooth background density in each panel

    Left panel:  prior p(z) density  (KDE of prior samples, same PCA if M>2)
    Right panel: aggregate posterior q(z|X) density + class-coloured scatter
    """
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA

    model.eval()
    M = model.prior.M if hasattr(model.prior, 'M') else model.prior.base_mean.shape[0]

    # Collect SAMPLES from q(z|x) for each test point + labels
    posterior_z_list, label_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= n_batches:
                break
            q = model.encoder(x.to(device))
            z = q.rsample()           # sample from q(z|x), not the mean
            posterior_z_list.append(z.cpu())
            label_list.append(y)

    posterior_z = torch.cat(posterior_z_list, dim=0).numpy()   # (N, M)
    labels      = torch.cat(label_list,       dim=0).numpy()   # (N,)

    # Sample same number of points from prior
    prior_z = _sample_prior(model, posterior_z.shape[0])       # (N, M)

    # PCA projection when M > 2
    if M > 2:
        pca = PCA(n_components=2)
        # Fit PCA on posterior samples (captures most variance of the data)
        posterior_2d = pca.fit_transform(posterior_z)
        prior_2d     = pca.transform(prior_z)
        xlabel = "PC 1"
        ylabel = "PC 2"
        subtitle = f"(PCA projection from {M}D)"
    else:
        posterior_2d = posterior_z
        prior_2d     = prior_z
        xlabel = "z₁"
        ylabel = "z₂"
        subtitle = ""

    # Shared axis limits (union of both)
    all_2d = np.concatenate([prior_2d, posterior_2d], axis=0)
    pad    = 1.0
    xlim   = (all_2d[:, 0].min() - pad, all_2d[:, 0].max() + pad)
    ylim   = (all_2d[:, 1].min() - pad, all_2d[:, 1].max() + pad)

    # KDE on the 2D projections
    res      = 200
    xs       = np.linspace(xlim[0], xlim[1], res)
    ys       = np.linspace(ylim[0], ylim[1], res)
    gx, gy   = np.meshgrid(xs, ys)
    grid_pts = np.vstack([gx.ravel(), gy.ravel()])

    density_prior = gaussian_kde(prior_2d.T)(grid_pts).reshape(res, res)
    density_post  = gaussian_kde(posterior_2d.T)(grid_pts).reshape(res, res)

    # Plot
    prior_label = PRIOR_LABELS[prior_name]
    tab10       = plt.cm.get_cmap("tab10", 10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    title = f"{prior_label} Prior: Prior vs Aggregate Posterior"
    if subtitle:
        title += f"  {subtitle}"
    fig.suptitle(title, fontsize=13)

    # Left: prior KDE
    ax = axes[0]
    # 1. Capture the contourf object as 'cp'
    cp = ax.contourf(gx, gy, density_prior, levels=30, cmap="Blues")
    ax.contour( gx, gy, density_prior, levels=6,  colors="white",
                linewidths=0.6, alpha=0.6)
    
    # 2. Add the colorbar using 'cp'
    cbar = fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Density', rotation=270, labelpad=15)
    
    ax.set_title("Prior p(z)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(xlim);     ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    # Right: posterior KDE + class-coloured scatter
    ax = axes[1]
    ax.contourf(gx, gy, density_post, levels=30, cmap="Reds", alpha=0.55)

    for digit in range(10):
        mask = labels == digit
        if mask.sum() == 0:
            continue
        ax.scatter(
            posterior_2d[mask, 0], posterior_2d[mask, 1],
            s=5, alpha=0.5, color=tab10(digit),
            label=str(digit), linewidths=0,
        )

    ax.set_title("Aggregate posterior q(z|x)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(xlim);     ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(title="digit", fontsize=7, title_fontsize=8,
              markerscale=2.5, loc="upper right",
              framealpha=0.75, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Plot saved -> {save_path}")
    plt.show()


def plot_samples(model, device, prior_name, save_path, n_samples=64):
    """
    Draw n_samples from p(z) -> p(x|z) and display as an 8-column grid.
    Uses the decoder mean (soft probabilities) for cleaner visuals.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model.prior, GaussianPrior):
            z = model.prior().sample(torch.Size([n_samples])).to(device)
        else:
            z = model.prior.sample(n_samples).to(device)
        samples = model.decoder(z).mean.cpu()   # (N, 28, 28)

    ncols = 8
    nrows = n_samples // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    fig.suptitle(f"{PRIOR_LABELS[prior_name]} Prior -- Generated Samples", fontsize=12)

    for idx, ax in enumerate(axes.flat):
        ax.imshow(samples[idx], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Samples saved -> {save_path}")
    plt.show()


# Build model

def build_model(prior_name, M, device, flow_layers=4, flow_hidden=64):
    """
    Architecture (identical for all priors):
      Encoder: Flatten -> Linear(784,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,2M)
               Output split into mean and log-std for q(z|x).
      Decoder: Linear(M,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,784) -> Unflatten(28,28)
               Outputs logits for a Bernoulli distribution over 28x28 pixels.
    """
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )
    if prior_name == "gaussian":
        prior = GaussianPrior(M)
    elif prior_name == "mog":
        prior = MoGPrior(M, K=10)
    elif prior_name == "flow":
        prior = FlowPrior(M, num_layers=flow_layers, hidden_dim=flow_hidden)
    else:
        raise ValueError(f"Unknown prior: {prior_name}")

    return VAE(prior, BernoulliDecoder(decoder_net), Encoder(encoder_net)).to(device)


# Main

if __name__ == "__main__":
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(description="VAE -- Part A (Gaussian + MoG priors)")
    parser.add_argument("mode", choices=["train", "evaluate", "sample", "multirun"],
                        help="train: single run | evaluate: load & eval | sample: generate images | multirun: N runs -> mean+/-std")
    parser.add_argument("--prior",      type=str,   default="gaussian", choices=["gaussian", "mog", "flow"])
    parser.add_argument("--model",      type=str,   default="model.pt",
                        help="Path to save/load model weights (used in train + evaluate)")
    parser.add_argument("--plot",       type=str,   default=None,
                        help="Output path for prior/posterior plot (default: <prior>_prior_posterior.png)")
    parser.add_argument("--samples",    type=str,   default=None,
                        help="Output path for samples image (default: <prior>_samples.png)")
    parser.add_argument("--device",     type=str,   default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--latent-dim", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--runs",       type=int,   default=5,
                        help="Number of independent training runs for multirun mode (default: 5)")
    parser.add_argument("--flow-layers", type=int,  default=4,
                        help="Number of coupling layers in the Flow prior (default: 4)")
    parser.add_argument("--flow-hidden", type=int,  default=64,
                        help="Hidden dim of scale/translation nets in Flow prior (default: 64)")
    args = parser.parse_args()

    if args.plot    is None: args.plot    = f"{args.prior}_prior_posterior.png"
    if args.samples is None: args.samples = f"{args.prior}_samples.png"

    print("# Options")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")

    device = torch.device(args.device)
    M      = args.latent_dim

    # Binarized MNIST
    binarize  = transforms.Lambda(lambda x: (x > 0.5).float().squeeze())
    transform = transforms.Compose([transforms.ToTensor(), binarize])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=True,  download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False,
    )

    # Single training run
    if args.mode == "train":
        model     = build_model(args.prior, M, device, args.flow_layers, args.flow_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, device)
        torch.save(model.state_dict(), args.model)
        print(f"[OK] Model saved -> {args.model}")

        elbo = evaluate_elbo(model, test_loader, device)
        print(f"\n[Test ELBO] {elbo:.4f}")

        plot_prior_vs_posterior(model, test_loader, device, args.prior, args.plot)
        plot_samples(model, device, args.prior, args.samples)

    # Sample from a saved model 
    elif args.mode == "sample":
        from torchvision.utils import save_image
        model = build_model(args.prior, M, device, args.flow_layers, args.flow_hidden)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)
        print(f"[OK] 64 samples saved -> {args.samples}")

    # Evaluate a saved model
    elif args.mode == "evaluate":
        model = build_model(args.prior, M, device, args.flow_layers, args.flow_hidden)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        elbo = evaluate_elbo(model, test_loader, device)
        print(f"\n[Test ELBO] {elbo:.4f}")

        plot_prior_vs_posterior(model, test_loader, device, args.prior, args.plot)
        plot_samples(model, device, args.prior, args.samples)

    # Multi-run: train N times, report mean +/- std
    elif args.mode == "multirun":
        multirun_evaluation(
            prior_name   = args.prior,
            M            = M,
            train_loader = train_loader,
            test_loader  = test_loader,
            epochs       = args.epochs,
            lr           = args.lr,
            device       = device,
            n_runs       = args.runs,
            flow_layers  = args.flow_layers,
            flow_hidden  = args.flow_hidden,
        )