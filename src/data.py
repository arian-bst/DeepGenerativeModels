import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .config import K, batch_size, train_subset

# -----------------------
# Data
# -----------------------
def quantize(x):
    return (x * (K - 1)).long()

mnist_tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(quantize),
])

def emnist_fix(img):
    img = transforms.functional.rotate(img, -90)
    img = transforms.functional.hflip(img)
    return img

emnist_tfs = transforms.Compose([
    transforms.Lambda(emnist_fix),
    transforms.ToTensor(),
    transforms.Lambda(quantize),
])

mnist = datasets.MNIST("./data", train=True, download=True, transform=mnist_tfs)
emnist = datasets.EMNIST("./data", split="letters", train=True, download=True, transform=emnist_tfs)

mnist_loader = DataLoader(torch.utils.data.Subset(mnist, range(train_subset)),
                          batch_size=batch_size, shuffle=True, drop_last=True)
emnist_loader = DataLoader(torch.utils.data.Subset(emnist, range(train_subset)),
                           batch_size=batch_size, shuffle=True, drop_last=True)

mnist_iter = iter(mnist_loader)
emnist_iter = iter(emnist_loader)

def next_batch(it, loader):
    try:
        x, y = next(it)
    except StopIteration:
        it = iter(loader)
        x, y = next(it)
    return it, x, y