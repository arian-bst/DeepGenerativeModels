import torch

# -----------------------
# Device / Hyperparams
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

K = 16
num_steps = 80
batch_size = 64

lr = 1e-4
ipf_iters = 2000

eps_keep = 0.9
eps = 1.0 - eps_keep

train_subset = 20000

# Endpoint GAN
d_lr = 2e-4
lambda_gan = 0.05            # weight of adversarial endpoint loss added to each M-projection loss
d_steps_per_iter = 1        # discriminator updates per outer iteration
label_smooth_real = 0.0     # e.g. 0.1 -> real labels become 0.9

# Gumbel-Softmax
# gumbel_tau = 1.0            # consider annealing 2.0 -> 0.5
gan_rollout_steps = 30      # IMPORTANT for memory; start 20-50 (not 200)

# IPF memory control
chunk_size = 10             # truncated backprop chunk (5-20)
# if still OOM, try chunk_size=5 or batch_size=32

torch.manual_seed(0)

plot_every = 20