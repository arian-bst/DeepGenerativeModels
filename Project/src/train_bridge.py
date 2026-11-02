import torch
import torch.nn as nn

from .config import device, K, num_steps, lr, ipf_iters, eps, d_lr, lambda_gan, d_steps_per_iter, chunk_size, plot_every
from .data import mnist_loader, emnist_loader, mnist_iter, emnist_iter, next_batch
from .models import DiscreteUNet, EndpointDiscriminator
from .helpers import tokens_to_onehot
from .kernels import forward_pairs_from_p, backward_pairs_from_q, lazy_logprob_step
from .gan_utils import rollout_forward_soft, rollout_backward_soft, d_loss_hinge, g_adv_loss_hinge, r1_penalty
from .viz import plot_current_bridge

# Instantiate
p_theta = DiscreteUNet(K, num_steps=num_steps).to(device)
q_phi   = DiscreteUNet(K, num_steps=num_steps).to(device)
D_T     = EndpointDiscriminator(K).to(device)  # EMNIST discriminator at time T
D_0     = EndpointDiscriminator(K).to(device)  # MNIST discriminator at time 0

opt_p = torch.optim.AdamW(p_theta.parameters(), lr=lr)
opt_q = torch.optim.AdamW(q_phi.parameters(), lr=lr)
opt_dT = torch.optim.AdamW(D_T.parameters(), lr=d_lr)
opt_d0 = torch.optim.AdamW(D_0.parameters(), lr=d_lr)

bce_logits = nn.BCEWithLogitsLoss()

# ============================================================
# Training
# ============================================================
mnist_iter_local = mnist_iter
emnist_iter_local = emnist_iter

mnist_iter_local, emnist_iter_local = plot_current_bridge(it=0, p_theta=p_theta, q_phi=q_phi,
                                                          mnist_iter=mnist_iter_local, emnist_iter=emnist_iter_local)

p_theta.train()
q_phi.train()
D_T.train()
D_0.train()

for it in range(1, ipf_iters + 1):

    # -----------------------------
    # Schedules (KEY FIX)
    # -----------------------------
    # Gumbel temperature schedule (smooth -> sharper)
    gumbel_tau = max(0.7, 2.0 - 1.3 * (it / 1000))

    # GAN rollout length schedule (match true endpoint progressively)
    gan_rollout_steps = 80

    # GAN weight warmup
    warm = min(1.0, it / 300)

    decay = 1.0 if it < 150 else max(0.3, 1.0 - (it - 150) / 400)

    lambda_gan_eff = lambda_gan * warm * decay


    # R1 penalty weight (you can tune 0.1, 1.0, 5.0)
    r1_gamma = 1.0

    # -----------------------------
    # (1) M-projection: Fit Q using samples from P (HARD pairs)
    #     + adversarial endpoint on x0 from Q (SOFT/Gumbel rollout)
    # -----------------------------
    mnist_iter_local, x0, _ = next_batch(mnist_iter_local, mnist_loader)
    x0 = x0.to(device)

    opt_q.zero_grad(set_to_none=True)
    loss_q_sum = 0.0
    steps_in_chunk = 0

    for t, x_t, x_tp1 in forward_pairs_from_p(p_theta, x0, num_steps, eps, K):
        # Q predicts x_t given x_{t+1}
        loss_step = lazy_logprob_step(q_phi, x_tp1, x_t, t, eps, K) / num_steps
        loss_step.backward()
        loss_q_sum += float(loss_step.item())
        steps_in_chunk += 1

        if steps_in_chunk % chunk_size == 0:
            opt_q.step()
            opt_q.zero_grad(set_to_none=True)

    if steps_in_chunk % chunk_size != 0:
        opt_q.step()

    loss_q = loss_q_sum

    # GAN generator loss for Q (separate backward+step for bounded memory)
    emnist_iter_local, xT_for_q_adv, _ = next_batch(emnist_iter_local, emnist_loader)
    xT_for_q_adv = xT_for_q_adv.to(device)

    x0_soft_from_q = rollout_backward_soft(
        q_phi, xT_for_q_adv,
        steps=gan_rollout_steps,
        eps=eps, K=K, tau=gumbel_tau
    )
    loss_q_adv = g_adv_loss_hinge(D_0, x0_soft_from_q)

    opt_q.zero_grad(set_to_none=True)
    (lambda_gan_eff * loss_q_adv).backward()
    opt_q.step()

    # -----------------------------
    # (2) Discriminator update(s) with R1 penalty (SOFT fakes)
    # -----------------------------
    for _ in range(d_steps_per_iter):

        # =====================================================
        # D_T: real EMNIST vs fake xT from P_theta rollout (SOFT)
        # =====================================================
        emnist_iter_local, xT_real, _ = next_batch(emnist_iter_local, emnist_loader)
        xT_real = xT_real.to(device)  # tokens (B,1,H,W)

        mnist_iter_local, x0_for_p_fake, _ = next_batch(mnist_iter_local, mnist_loader)
        x0_for_p_fake = x0_for_p_fake.to(device)  # tokens

        with torch.no_grad():
            xT_fake_soft = rollout_forward_soft(
                p_theta, x0_for_p_fake,
                steps=gan_rollout_steps,
                eps=eps, K=K, tau=gumbel_tau
            )

        # R1 needs grad w.r.t. D input -> provide real as one-hot float
        xT_real_oh = tokens_to_onehot(xT_real, K)

        opt_dT.zero_grad(set_to_none=True)
        loss_dT = d_loss_hinge(D_T, xT_real, xT_fake_soft) + r1_gamma * r1_penalty(D_T, xT_real_oh)
        loss_dT.backward()
        opt_dT.step()

        # =====================================================
        # D_0: real MNIST vs fake x0 from Q_phi rollout (SOFT)
        # =====================================================
        mnist_iter_local, x0_real, _ = next_batch(mnist_iter_local, mnist_loader)
        x0_real = x0_real.to(device)  # tokens

        emnist_iter_local, xT_for_q_fake, _ = next_batch(emnist_iter_local, emnist_loader)
        xT_for_q_fake = xT_for_q_fake.to(device)  # tokens

        with torch.no_grad():
            x0_fake_soft = rollout_backward_soft(
                q_phi, xT_for_q_fake,
                steps=gan_rollout_steps,
                eps=eps, K=K, tau=gumbel_tau
            )

        x0_real_oh = tokens_to_onehot(x0_real, K)

        opt_d0.zero_grad(set_to_none=True)
        loss_d0 = d_loss_hinge(D_0, x0_real, x0_fake_soft) + r1_gamma * r1_penalty(D_0, x0_real_oh)
        loss_d0.backward()
        opt_d0.step()

    # -----------------------------
    # (3) M-projection: Fit P using samples from Q (HARD pairs)
    #     + adversarial endpoint on xT from P (SOFT/Gumbel rollout)
    # -----------------------------
    emnist_iter_local, xT, _ = next_batch(emnist_iter_local, emnist_loader)
    xT = xT.to(device)

    opt_p.zero_grad(set_to_none=True)
    loss_p_sum = 0.0
    steps_in_chunk = 0

    for t, x_t, x_tp1 in backward_pairs_from_q(q_phi, xT, num_steps, eps, K):
        # P predicts x_{t+1} given x_t
        loss_step = lazy_logprob_step(p_theta, x_t, x_tp1, t, eps, K) / num_steps
        loss_step.backward()
        loss_p_sum += float(loss_step.item())
        steps_in_chunk += 1

        if steps_in_chunk % chunk_size == 0:
            opt_p.step()
            opt_p.zero_grad(set_to_none=True)

    if steps_in_chunk % chunk_size != 0:
        opt_p.step()

    loss_p = loss_p_sum

    # GAN generator loss for P (separate backward+step)
    mnist_iter_local, x0_for_p_adv, _ = next_batch(mnist_iter_local, mnist_loader)
    x0_for_p_adv = x0_for_p_adv.to(device)

    xT_soft_from_p = rollout_forward_soft(
        p_theta, x0_for_p_adv,
        steps=gan_rollout_steps,
        eps=eps, K=K, tau=gumbel_tau
    )
    loss_p_adv = g_adv_loss_hinge(D_T, xT_soft_from_p)

    opt_p.zero_grad(set_to_none=True)
    (lambda_gan_eff * loss_p_adv).backward()
    opt_p.step()

    # -----------------------------
    # Logging
    # -----------------------------
    print(
        f"Iter {it}: "
        f"loss_q={loss_q:.4f} | loss_q_adv={float(loss_q_adv.item()):.4f} | "
        f"loss_p={loss_p:.4f} | loss_p_adv={float(loss_p_adv.item()):.4f} | "
        f"loss_dT={float(loss_dT.item()):.4f} | loss_d0={float(loss_d0.item()):.4f} | "
        f"tau={gumbel_tau:.3f} | gan_steps={gan_rollout_steps} | lambda_gan_eff={lambda_gan_eff:.4f}"
    )

    # -----------------------------
    # Plot every N iterations
    # -----------------------------
    if it % plot_every == 0:
        mnist_iter_local, emnist_iter_local = plot_current_bridge(it, p_theta, q_phi, mnist_iter_local, emnist_iter_local)