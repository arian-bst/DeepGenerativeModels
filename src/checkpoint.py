import torch
from .config import device

def load_checkpoint(path, p_theta, q_phi, D_T=None, D_0=None):
    ckpt = torch.load(path, map_location=device)

    # Common cases: either raw state_dicts or nested
    if isinstance(ckpt, dict) and "p_theta" in ckpt and "q_phi" in ckpt:
        p_theta.load_state_dict(ckpt["p_theta"])
        q_phi.load_state_dict(ckpt["q_phi"])
        if D_T is not None and "D_T" in ckpt:
            D_T.load_state_dict(ckpt["D_T"])
        if D_0 is not None and "D_0" in ckpt:
            D_0.load_state_dict(ckpt["D_0"])
    else:
        # fallback: if you saved only one model or a flat state_dict
        # (you can delete this branch if not needed)
        try:
            p_theta.load_state_dict(ckpt)
        except Exception:
            pass
        try:
            q_phi.load_state_dict(ckpt)
        except Exception:
            pass

    p_theta.eval()
    q_phi.eval()
    if D_T is not None: D_T.eval()
    if D_0 is not None: D_0.eval()

    return ckpt