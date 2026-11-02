from .config import K, num_steps, device
from .models import DiscreteUNet
from .checkpoint import load_checkpoint
from .sample_backward import sample_endpoints_b

import src.sample_backward as sb

sb.p_theta = DiscreteUNet(K, num_steps=num_steps).to(device)
sb.q_phi   = DiscreteUNet(K, num_steps=num_steps).to(device)

load_checkpoint("bridge_checkpoint/bridge.pt", sb.p_theta, sb.q_phi)

sample_endpoints_b(n=50)