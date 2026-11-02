import torch
import torch.nn.functional as F

from .helpers import tokens_to_onehot

# -----------------------
# Differentiable (ST Gumbel) lazy step for GAN rollouts
# -----------------------
def gumbel_sample_onehot_st(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """
    logits: (B,K,H,W) -> returns (B,K,H,W) straight-through one-hot samples
    """
    B, Kc, H, W = logits.shape
    flat = logits.permute(0, 2, 3, 1).reshape(-1, Kc)
    y = F.gumbel_softmax(flat, tau=tau, hard=True, dim=1)  # ST
    y = y.view(B, H, W, Kc).permute(0, 3, 1, 2).contiguous()
    return y

def lazy_sample_step_soft_st(model, x_soft, t_scalar, eps, K, tau):
    """
    x_soft: (B,K,H,W) (soft or one-hot)
    returns x_next_soft: (B,K,H,W) with ST gradients through proposal logits
    """
    B = x_soft.size(0)
    t = torch.full((B,), t_scalar, device=x_soft.device, dtype=torch.long)
    logits = model(x_soft, t)            # (B,K,H,W)
    proposal_onehot = gumbel_sample_onehot_st(logits, tau=tau)

    mask = (torch.rand((B, 1, x_soft.size(2), x_soft.size(3)), device=x_soft.device) < eps).float()
    x_next = (1.0 - mask) * x_soft + mask * proposal_onehot
    return x_next

def rollout_forward_soft(p_model, x0_tokens, steps, eps, K, tau):
    x = tokens_to_onehot(x0_tokens, K)
    for t in range(steps):
        x = lazy_sample_step_soft_st(p_model, x, t, eps, K, tau)
    return x

def rollout_backward_soft(q_model, xT_tokens, steps, eps, K, tau):
    x = tokens_to_onehot(xT_tokens, K)
    for t in reversed(range(steps)):
        x = lazy_sample_step_soft_st(q_model, x, t, eps, K, tau)
    return x

def r1_penalty(D, x_real):
    x_real = x_real.detach().requires_grad_(True)
    logits = D(x_real)
    grad = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=x_real,
        create_graph=True
    )[0]
    return 0.5 * (grad.pow(2).reshape(grad.size(0), -1).sum(dim=1)).mean()

# -----------------------
# Hinge GAN losses (more stable than BCE)
# -----------------------
def d_loss_hinge(D, x_real, x_fake):
    """
    D outputs realness scores (logits). Hinge loss:
      L_D = E[relu(1 - D(real))] + E[relu(1 + D(fake))]
    """
    real_scores = D(x_real)
    fake_scores = D(x_fake.detach())  # detach fakes for D update

    loss_real = F.relu(1.0 - real_scores).mean()
    loss_fake = F.relu(1.0 + fake_scores).mean()
    return loss_real + loss_fake

def g_adv_loss_hinge(D, x_fake):
    """
    Generator hinge loss:
      L_G = -E[D(fake)]
    """
    fake_scores = D(x_fake)
    return -fake_scores.mean()