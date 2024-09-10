import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl


def torch_L2normalize(x, d=1):
    eps = 1e-6
    norm = x ** 2
    norm = norm.sum(dim=d, keepdim=True) + eps
    norm = norm ** (0.5)
    return (x / norm)
