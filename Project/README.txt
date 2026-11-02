# DiscreteBridge Project (MNIST ↔ EMNIST)

This repository contains our implementation of the approach described in our [final project report (PDF)](reports/DiscreteBridge_Report.pdf), tested on MNIST digits and EMNIST letters. It trains a discrete-time bridge between the two datasets using IPF and endpoint adversarial losses.

The code is split into modules in `src/`:
- `config.py`: hyperparameters
- `data.py`: MNIST/EMNIST loading + iterators
- `helpers.py`: token ↔ one-hot conversions
- `models.py`: U-Net kernels + endpoint discriminators
- `kernels.py`: lazy discrete kernel + IPF pair generation
- `gan_utils.py`: Gumbel-ST rollout + hinge GAN losses + R1 penalty
- `viz.py`: trajectory visualization helpers
- `train_bridge.py`: full training loop (slow)

## Setup

```bash
pip install -r requirements.txt

## Running the Code

All scripts are located inside the `src/` folder and should be executed from the project root directory using Python module mode.

Example:
python -m src.reproduce_forward


## Reproducing Results

To reproduce the forward translation results (MNIST → EMNIST):
python -m src.reproduce_forward

This will:
- load the pretrained bridge models
- generate samples starting from MNIST inputs
- save the output grids inside the `outputs/` folder


To reproduce the backward translation results (EMNIST → MNIST):
python -m src.reproduce_backward

This will:
- load the pretrained models
- generate reverse bridge samples
- save the resulting grids into `outputs/`.


## Training

The full training procedure can be run using:
python -m src.train_bridge

Note:
- Training is expensive and may take a long time.


## Project Overview
Images are discretized into K-level pixel tokens and evolved through a lazy stochastic kernel.  
Two neural networks learn forward and backward transition dynamics between MNIST and EMNIST domains using Iterative Proportional Fitting (IPF).

Endpoint adversarial losses are added to encourage realistic endpoint distributions, implemented using hinge GAN objectives with R1 regularization.
