import torch
import matplotlib.pyplot as plt
import numpy as np
import math

from .config import device, num_steps, eps, K
from .kernels import lazy_sample_step_hard
from .data import next_batch, mnist_loader, emnist_loader

# -----------------------
# Visualization (HARD rollouts)
# -----------------------
@torch.no_grad()
def sample_forward_trajectory(p_model, x0, num_steps, eps, K):
    xs = [x0]
    x = x0
    for t in range(num_steps):
        x = lazy_sample_step_hard(p_model, x, t, eps, K)
        xs.append(x)
    return xs

@torch.no_grad()
def sample_backward_trajectory(q_model, xT, num_steps, eps, K):
    xs = [xT]
    x = xT
    for t in reversed(range(num_steps)):
        x = lazy_sample_step_hard(q_model, x, t, eps, K)
        xs.append(x)
    return xs

@torch.no_grad()
def show_forward_bridge(p_model, x0, num_cols=10):
    p_model.eval()
    x0 = x0.to(device)
    traj = sample_forward_trajectory(p_model, x0, num_steps, eps, K)

    idx = (torch.logspace(0, math.log10(num_steps + 1), num_cols) - 1).long()
    idx = torch.unique(idx.clamp(0, num_steps))
    imgs = [traj[i].cpu() for i in idx.tolist()]

    B = x0.size(0)
    n = min(B, 4)
    fig, axes = plt.subplots(n, len(imgs), figsize=(18, 2*n))
    fig.suptitle("Forward P_theta trajectory (MNIST -> EMNIST)", fontsize=14)

    for i in range(n):
        for j, im in enumerate(imgs):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(im[i].squeeze(), cmap="gray", vmin=0, vmax=K-1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={idx[j].item()}")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def show_backward_bridge(q_model, xT, num_cols=10):
    q_model.eval()
    xT = xT.to(device)
    traj = sample_backward_trajectory(q_model, xT, num_steps, eps, K)
    traj = list(reversed(traj))

    idx = (torch.logspace(0, math.log10(num_steps + 1), num_cols) - 1).long()
    idx = torch.unique(idx.clamp(0, num_steps))
    imgs = [traj[i].cpu() for i in idx.tolist()]

    B = xT.size(0)
    n = min(B, 4)
    fig, axes = plt.subplots(n, len(imgs), figsize=(18, 2*n))
    fig.suptitle("Backward Q_phi trajectory (EMNIST -> MNIST)", fontsize=14)

    for i in range(n):
        for j, im in enumerate(imgs):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(im[i].squeeze(), cmap="gray", vmin=0, vmax=K-1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={idx[j].item()}")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_current_bridge(it, p_theta, q_phi, mnist_iter, emnist_iter):
    print(f"\n================ Iter {it} ================")

    mnist_iter, x0_demo, _ = next_batch(mnist_iter, mnist_loader)
    print("Forward P_theta (MNIST -> EMNIST)")
    show_forward_bridge(p_theta, x0_demo[:4], num_cols=10)

    emnist_iter, xT_demo, _ = next_batch(emnist_iter, emnist_loader)
    print("Backward Q_phi (EMNIST -> MNIST)")
    show_backward_bridge(q_phi, xT_demo[:4], num_cols=10)

    return mnist_iter, emnist_iter