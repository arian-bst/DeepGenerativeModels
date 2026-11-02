from .config import K, num_steps, device
from .models import DiscreteUNet
from .checkpoint import load_checkpoint
from .sample_forward import sample_endpoints_f  # uses global p_theta inside that module

import src.sample_forward as sf

# Recreate the same globals expected by sample_forward.py
sf.p_theta = DiscreteUNet(K, num_steps=num_steps).to(device)
sf.q_phi   = DiscreteUNet(K, num_steps=num_steps).to(device)

load_checkpoint("bridge_checkpoint/bridge.pt", sf.p_theta, sf.q_phi)

sample_endpoints_f(n=50)