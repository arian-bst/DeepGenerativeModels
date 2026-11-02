import torch
import torch.nn.functional as F

# -----------------------
# Discrete lazy kernel (HARD sampling for IPF data generation)
# -----------------------
@torch.no_grad()
def lazy_sample_step_hard(model, x_tokens, t_scalar, eps, K):
    B = x_tokens.size(0)
    t = torch.full((B,), t_scalar, device=x_tokens.device, dtype=torch.long)
    logits = model(x_tokens, t)               # (B,K,H,W)
    probs = F.softmax(logits, dim=1)          # (B,K,H,W)

    p_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)
    proposal = torch.multinomial(p_flat.clamp_min(1e-8), 1).view(B, 1, x_tokens.size(2), x_tokens.size(3))

    update = (torch.rand_like(x_tokens.float()) < eps)
    x_next = torch.where(update, proposal, x_tokens)
    return x_next

def lazy_logprob_step(model, x_cond_tokens, x_target_tokens, t_scalar, eps, K):
    B, _, H, W = x_cond_tokens.shape
    t = torch.full((B,), t_scalar, device=x_cond_tokens.device, dtype=torch.long)
    logits = model(x_cond_tokens, t)          # (B,K,H,W)
    probs = F.softmax(logits, dim=1)

    tgt = x_target_tokens.squeeze(1).long()   # (B,H,W)
    p_nn = probs.gather(1, tgt.unsqueeze(1)).squeeze(1)

    same = (tgt == x_cond_tokens.squeeze(1).long()).float()
    p = (1.0 - eps) * same + eps * p_nn

    nll = -(torch.log(p.clamp_min(1e-8))).mean()
    return nll

# -----------------------
# IPF pair generators (HARD, no-grad)
# -----------------------
@torch.no_grad()
def forward_pairs_from_p(p_model, x0, num_steps, eps, K):
    x = x0
    for t in range(num_steps):
        x_next = lazy_sample_step_hard(p_model, x, t, eps, K)
        yield t, x, x_next
        x = x_next

@torch.no_grad()
def backward_pairs_from_q(q_model, xT, num_steps, eps, K):
    x = xT
    pairs = []
    for t in reversed(range(num_steps)):
        x_t = lazy_sample_step_hard(q_model, x, t, eps, K)
        pairs.append((t, x_t, x))
        x = x_t
    for item in reversed(pairs):
        yield item