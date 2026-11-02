import torch
import torch.nn.functional as F

# -----------------------
# Helpers: token <-> onehot
# -----------------------
def tokens_to_onehot(x_tokens: torch.Tensor, K: int) -> torch.Tensor:
    # x_tokens: (B,1,H,W) long -> (B,K,H,W) float
    return F.one_hot(x_tokens.squeeze(1).long(), K).float().permute(0, 3, 1, 2)

def onehot_to_tokens(x_onehot: torch.Tensor) -> torch.Tensor:
    # x_onehot: (B,K,H,W) -> (B,1,H,W) long
    return x_onehot.argmax(dim=1, keepdim=True).long()