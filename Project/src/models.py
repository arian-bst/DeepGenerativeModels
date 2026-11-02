import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import tokens_to_onehot

# -----------------------
# Models
# -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x + residual

class DiscreteUNet(nn.Module):
    """
    Accepts either:
      - tokens: (B,1,H,W) long
      - soft/onehot: (B,K,H,W) float
    Returns logits: (B,K,H,W)
    """
    def __init__(self, K, num_steps):
        super().__init__()
        self.K = K
        self.time_emb = nn.Sequential(
            nn.Embedding(num_steps, 128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.time_proj_down1 = nn.Linear(128, 64)
        self.time_proj_down2 = nn.Linear(128, 128)
        self.time_proj_mid   = nn.Linear(128, 128)
        self.time_proj_up1   = nn.Linear(128, 64)

        self.down1 = ResidualBlock(K, 64)
        self.downsample1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(64, 128)
        self.downsample2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)

        self.mid = ResidualBlock(128, 128)

        self.upsample1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(128 + 128, 64)
        self.upsample2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(64 + 64, K)

    def forward(self, x, t):
        if x.dtype in (torch.int64, torch.int32, torch.int16) and x.dim() == 4 and x.size(1) == 1:
            x_in = tokens_to_onehot(x, self.K)  # (B,K,H,W)
        else:
            x_in = x  # (B,K,H,W) float

        t_emb_base = self.time_emb(t)
        t_emb_down1 = self.time_proj_down1(t_emb_base).view(-1, 64, 1, 1)
        t_emb_down2 = self.time_proj_down2(t_emb_base).view(-1, 128, 1, 1)
        t_emb_mid   = self.time_proj_mid(t_emb_base).view(-1, 128, 1, 1)
        t_emb_up1   = self.time_proj_up1(t_emb_base).view(-1, 64, 1, 1)

        d1 = self.down1(x_in) + t_emb_down1
        ds1 = self.downsample1(d1)
        d2 = self.down2(ds1) + t_emb_down2
        ds2 = self.downsample2(d2)

        m = self.mid(ds2) + t_emb_mid

        u1 = self.upsample1(m)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1) + t_emb_up1

        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        logits = self.up2(u2)
        return logits

class EndpointDiscriminator(nn.Module):
    """
    Accepts either:
      - tokens: (B,1,H,W) long
      - soft/onehot: (B,K,H,W) float
    Outputs: (B,) logits.
    """
    def __init__(self, K: int):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Conv2d(K, 64, 4, stride=2, padding=1),   # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 14->7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1)
        )

    def forward(self, x):
        if x.dtype in (torch.int64, torch.int32, torch.int16) and x.dim() == 4 and x.size(1) == 1:
            x_in = tokens_to_onehot(x, self.K)
        else:
            x_in = x
        return self.net(x_in).squeeze(1)