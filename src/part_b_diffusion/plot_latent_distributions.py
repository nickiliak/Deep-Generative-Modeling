import os
import json
import time
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (saves to file instead of displaying)
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


def _sample_prior(vae, N):
    with torch.no_grad():
        # GaussianPrior exposes callable to return a distribution
        if hasattr(vae.prior, '__call__') or hasattr(vae.prior, 'forward'):
            try:
                return vae.prior().sample(torch.Size([N])).cpu().numpy()
            except Exception:
                return vae.prior.sample(N).cpu().numpy()
        else:
            return vae.prior.sample(N).cpu().numpy()


def collect_aggregate_posterior(vae, data_loader, device, n_samples=5000):
    """Collect samples from q(z|x) by encoding dataset examples.

    Returns numpy array shape (n_samples, M).
    """
    zs = []
    with torch.no_grad():
        for x in data_loader:
            # data_loader used for VAE train/test returns (batch, 28,28) or (batch,1,28,28)
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            q = vae.encoder(x)
            z = q.rsample().cpu()
            zs.append(z)
            if sum(z.shape[0] for z in zs) >= n_samples:
                break
    zs = torch.cat(zs, dim=0)[:n_samples].numpy()
    return zs


def collect_ddpm_latent_samples(ddpmlatent, n_samples=5000, device='cpu'):
    """Sample latent vectors from the DDPM component.

    Returns numpy array shape (n_samples, M).
    """
    with torch.no_grad():
        z = ddpmlatent.ddpm.sample((n_samples, ddpmlatent.latent_dim)).cpu().numpy()
    return z


def project_with_pca(reference, arrays):
    """Fit PCA on reference (n, M) and transform list of arrays."""
    M = reference.shape[1]
    if M > 2:
        pca = PCA(n_components=2)
        ref2 = pca.fit_transform(reference)
        out = [pca.transform(a) for a in arrays]
        return ref2, out
    else:
        # Already 2D
        return reference, arrays


def compare_latent_distributions(model, data_loader, device='cpu', n_samples=5000, save_dir='reports/outputs/latent_distributions'):
    """Collect prior, aggregate posterior, and DDPM latent samples, project to 2D and plot.

    Saves PNG and JSON stats to `save_dir` with timestamped filenames.
    
    Parameters:
    model: DDPMLatent model with vae and ddpm components
    data_loader: DataLoader for computing aggregate posterior
    device: Device to run on
    n_samples: Number of samples to collect for each distribution
    save_dir: Output directory (default: reports/outputs/latent_distributions)
    """
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.perf_counter()

    latent_dim = model.latent_dim

    # Collect aggregate posterior
    post_z = collect_aggregate_posterior(model.vae, data_loader, device, n_samples=n_samples)

    # Prior samples
    prior_z = _sample_prior(model.vae, post_z.shape[0])

    # DDPM latent samples
    ddpm_z = collect_ddpm_latent_samples(model, n_samples=post_z.shape[0], device=device)

    # PCA projection (fit on posterior)
    ref2d, (prior2d, post2d, ddpm2d) = project_with_pca(post_z, [prior_z, post_z, ddpm_z])

    # Axis limits for the first two panels (prior + aggregate posterior)
    pp_all2d = np.concatenate([prior2d, post2d], axis=0)
    lo_pct, hi_pct = 1.0, 99.0
    x_min, x_max = np.percentile(pp_all2d[:, 0], [lo_pct, hi_pct])
    y_min, y_max = np.percentile(pp_all2d[:, 1], [lo_pct, hi_pct])
    # Add small padding relative to interquartile range for breathing room
    x_pad = max(1.0, 0.05 * (x_max - x_min))
    y_pad = max(1.0, 0.05 * (y_max - y_min))
    xlim_pp = (x_min - x_pad, x_max + x_pad)
    ylim_pp = (y_min - y_pad, y_max + y_pad)

    # DDPM panel: compute its own limits so we don't compress its structure
    dd_x_min, dd_x_max = np.percentile(ddpm2d[:, 0], [lo_pct, hi_pct])
    dd_y_min, dd_y_max = np.percentile(ddpm2d[:, 1], [lo_pct, hi_pct])
    dd_x_pad = max(1.0, 0.05 * (dd_x_max - dd_x_min))
    dd_y_pad = max(1.0, 0.05 * (dd_y_max - dd_y_min))
    xlim_dd = (dd_x_min - dd_x_pad, dd_x_max + dd_x_pad)
    ylim_dd = (dd_y_min - dd_y_pad, dd_y_max + dd_y_pad)

    # KDE densities: build separate grids for prior/posterior (same) and DDPM (own)
    res = 200
    # grid for prior + posterior panels
    xs_pp = np.linspace(xlim_pp[0], xlim_pp[1], res)
    ys_pp = np.linspace(ylim_pp[0], ylim_pp[1], res)
    gx_pp, gy_pp = np.meshgrid(xs_pp, ys_pp)
    grid_pp = np.vstack([gx_pp.ravel(), gy_pp.ravel()])

    # grid for DDPM panel
    xs_dd = np.linspace(xlim_dd[0], xlim_dd[1], res)
    ys_dd = np.linspace(ylim_dd[0], ylim_dd[1], res)
    gx_dd, gy_dd = np.meshgrid(xs_dd, ys_dd)
    grid_dd = np.vstack([gx_dd.ravel(), gy_dd.ravel()])

    density_prior = gaussian_kde(prior2d.T)(grid_pp).reshape(res, res)
    density_post = gaussian_kde(post2d.T)(grid_pp).reshape(res, res)
    density_ddpm = gaussian_kde(ddpm2d.T)(grid_dd).reshape(res, res)

    grids = [(gx_pp, gy_pp), (gx_pp, gy_pp), (gx_dd, gy_dd)]

    # Plot three panels with improved visuals
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Prior p(z)", "Aggregate posterior q(z|x)", "Latent DDPM p_ddpm(z)"]
    densities = [density_prior, density_post, density_ddpm]
    points = [prior2d, post2d, ddpm2d]
    cmaps = ["Blues", "Reds", "Greens"]

    # Downsample scatter points for visibility (if large)
    def ds(pts, max_pts=2000):
        if pts.shape[0] > max_pts:
            idx = np.random.choice(pts.shape[0], max_pts, replace=False)
            return pts[idx]
        return pts

    for i, (ax, title, den, pts, cmap) in enumerate(zip(axes, titles, densities, points, cmaps)):
        gx, gy = grids[i]
        cp = ax.contourf(gx, gy, den, levels=30, cmap=cmap, alpha=0.9)
        ax.contour(gx, gy, den, levels=6, colors="white", linewidths=0.6, alpha=0.6)
        spts = ds(pts)
        ax.scatter(spts[:, 0], spts[:, 1], s=10, alpha=0.45, color="#222222", edgecolors='none')

        # Plot mean marker and 2-sigma covariance ellipse
        mu = np.mean(pts, axis=0)
        cov = np.cov(pts, rowvar=False)
        try:
            from matplotlib.patches import Ellipse
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            # 2-sigma ellipse
            width, height = 2 * np.sqrt(vals[:2]) * 2.0
            ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta,
                              edgecolor='k', facecolor='none', linewidth=1.0, alpha=0.9)
            if i in (0, 1):
                ax.add_patch(ellipse)
        except Exception:
            pass

        ax.plot(mu[0], mu[1], marker='x', color='k', markersize=6, mew=2)

        ax.set_title(title)
        # Apply prior/posterior limits to first two panels, DDPM own limits to last
        if i in (0, 1):
            ax.set_xlim(xlim_pp); ax.set_ylim(ylim_pp)
        else:
            ax.set_xlim(xlim_dd); ax.set_ylim(ylim_dd)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        # Add a small colorbar per panel
        cbar = fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(save_dir, f"comparison_{timestamp}.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    # Save statistics
    stats = {}
    for name, arr in zip(['prior', 'aggregate_posterior', 'ddpm'], [prior_z, post_z, ddpm_z]):
        stats[name] = {
            'mean': np.mean(arr, axis=0).tolist(),
            'cov': np.cov(arr, rowvar=False).tolist(),
            'n_samples': int(arr.shape[0])
        }

    stats['meta'] = {
        'latent_dim': int(latent_dim),
        'n_drawn': int(post_z.shape[0]),
        'pca_fitted_on': 'aggregate_posterior',
        'timestamp': timestamp,
        'elapsed_sec': time.perf_counter() - t0,
    }

    out_json = os.path.join(save_dir, f"stats_{timestamp}.json")
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Plot saved -> {out_png}")
    print(f"[OK] Stats saved -> {out_json}")
    return out_png, out_json


if __name__ == '__main__':
    print('This module provides `compare_latent_distributions(model, data_loader, ...)`.')
