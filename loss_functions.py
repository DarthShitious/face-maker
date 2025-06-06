import torch

def huber_loss(output, target, delta=1.0):
    """
    Huber Loss
    """
    diff = output - target
    abs_diff = torch.abs(diff)
    quadratic = torch.min(abs_diff, torch.tensor(delta, device=output.device))
    linear = abs_diff - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)

def mse_loss(output, target):
    """
    Mean Squared Error Loss
    """
    return torch.mean((output - target) ** 2)

def mae_loss(output, target):
    """
    Mean Absolute Error Loss
    """
    return torch.mean(torch.abs(output - target))

def kl_divergence(mu, logvar):
    """
    Kullback-Leibler Divergence Loss
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def covariance_loss(z):
    """
    z: Tensor of shape (batch_size, latent_dim)
    """
    # First center the latent vectors
    z_centered = z - z.mean(dim=0, keepdim=True)  # (batch_size, latent_dim)

    # Compute the covariance matrix (latent_dim x latent_dim)
    cov = (z_centered.T @ z_centered) / (z_centered.size(0) - 1)

    # Compute off-diagonal elements
    off_diagonal_mask = ~torch.eye(cov.size(0), dtype=bool, device=z.device)
    off_diag_cov = cov[off_diagonal_mask]

    # Penalize the squared off-diagonal entries
    loss = (off_diag_cov ** 2).mean()
    return loss


# def flow_loss(z, log_jac_det):
#     # Normalize both terms per element
#     B = z.size(0)
#     D = z[0].numel()
#     prior = torch.distributions.Normal(0, 1)

#     log_p_z = prior.log_prob(z).view(B, -1).sum(dim=1)
#     loss = (-log_p_z + log_jac_det) / D
#     return loss.mean()

def flow_loss_func(z, log_jac_det):
    # log_jac_det *= 0
    B = z.size(0)
    D = z[0].numel()

    # log p(z) under standard normal: -0.5 * (z^2 + log(2π))
    log_p_z = -0.5 * (z**2 + torch.log(torch.tensor(2 * torch.pi))).view(B, -1).sum(dim=1)

    # log_prob + log_det => full log-likelihood
    log_likelihood = log_p_z + log_jac_det  # shape [B]

    # Return negative log-likelihood per dim
    nll = -log_likelihood / D

    # penalize large ljd
    ljd_penalty = 0.0001 * (log_jac_det ** 2)

    # print("ljd penalty:      ",  ljd_penalty.mean().item())
    # print("log_p_z.mean():",     log_p_z.mean().item())
    # print("log_jac_det.mean():", log_jac_det.mean().item())
    # print("nll.mean():",         nll.mean().item())

    return nll.mean() + ljd_penalty.mean()



def recon_loss(original, recon):
    return torch.pow(original - recon, 2).mean()