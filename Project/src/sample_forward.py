import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import os

from .config import device, K, num_steps, eps
from .data import mnist_loader, emnist_loader, mnist_iter, emnist_iter, next_batch
from .models import DiscreteUNet
from .kernels import lazy_sample_step_hard

# Instantiate
p_theta = DiscreteUNet(K, num_steps=num_steps).to(device)
q_phi   = DiscreteUNet(K, num_steps=num_steps).to(device)

@torch.no_grad()
def sample_endpoints_f(n=50):
    global mnist_iter, emnist_iter

    p_theta.eval()
    q_phi.eval()

    # --------------------------------------------------
    # helper: collect N samples from loader
    # --------------------------------------------------
    def collect_tokens(n, loader_iter, loader):
        xs = []
        total = 0
        while total < n:
            loader_iter, x, _ = next_batch(loader_iter, loader)
            xs.append(x)
            total += x.size(0)
        xs = torch.cat(xs, dim=0)[:n]
        return loader_iter, xs

    # --------------------------------------------------
    # MNIST -> EMNIST (forward P)
    # --------------------------------------------------
    mnist_iter, x0 = collect_tokens(n, mnist_iter, mnist_loader)
    x0 = x0.to(device)

    x = x0.clone()
    for t in range(num_steps):
        x = lazy_sample_step_hard(p_theta, x, t, eps, K)
    xT = x


    # --------------------------------------------------
    # grid plotting
    # --------------------------------------------------
    def grid_show(imgs, title):
        imgs = imgs.cpu()
        cols = 10                  # fixed width grid
        rows = int(math.ceil(len(imgs) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.3, rows*1.3))
        fig.suptitle(title)

        axes = np.array(axes).reshape(-1)

        for i in range(rows*cols):
            axes[i].axis("off")
            if i < len(imgs):
                axes[i].imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=K-1)

        
        os.makedirs("outputs", exist_ok=True)

        plt.tight_layout()
        save_path = f"outputs/{title.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
        plt.close()

    # --------------------------------------------------
    # show results
    # --------------------------------------------------
    grid_show(x0,      "50 MNIST inputs")
    grid_show(xT,      "50 MNIST → EMNIST samples (P_theta)")


sample_endpoints_f(n=50)