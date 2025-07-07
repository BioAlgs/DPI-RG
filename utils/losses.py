import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.autograd as autograd
from torch.autograd import grad
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Helper function to get the device
def get_device(tensor):
    return tensor.device

# loss function for f update:
def f_loss(netI, netG, netf, z, fake_z):
    post_z = netI(netG(z))
    losses = netf(fake_z) - netf(post_z)
    return losses.mean()

# loss function for G and I update:
def GI_loss(netI, netG, netf, z, fake_z, p=2):
    post_z = netI(netG(z))
    n_dim = len(post_z.shape)
    dim = list(range(1, n_dim))
    # distance = torch.dist(real_data, post_data, p=p)
    # sz = 1
    l2 = torch.sqrt(torch.sum((z-post_z)**2, dim=dim))
    # sz = real_data.shape[0]
    losses = l2 + netf(post_z) - netf(fake_z)
    return losses.mean()

# reconstruction for z
def rec_z(netG, netI, z):
    z_t = netI(netG(z))
    n_dim = len(z_t.shape)
    dim = list(range(1, n_dim))
    l2 = torch.sqrt(torch.sum((z-z_t)**2, dim=dim))
    return l2.mean()

# primal loss function
def primal(netI, netG, netf, z, p=2):
    post_z = netI(netG(z))
    n_dim = len(post_z.shape)
    dim = list(range(1, n_dim))
    l2 = torch.sqrt(torch.sum((z-post_z)**2, dim=dim))
    return l2.mean()


# dual loss function
def dual(netI, netG, netf, z, fake_z):
    losses = netf(z) - netf(fake_z)
    return losses.mean()

# *** Penalty Terms ***
# Gradient Penalty
# gradient penalty function
def _gradient_penalty(x, y, f):
    device = get_device(x)
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(device)
    z = x + alpha * (y - x)
    # gradient penalty
    z = Variable(z, requires_grad=True).to(device)
    o = f(z)
    g = grad(o, z, grad_outputs=(torch.ones(o.size())).to(device), create_graph=True)[0].view(z.size(0), -1)
    # g = grad(o, z, grad_outputs=(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp

# gradient penalty for netf
def gradient_penalty_f(x, z, netf, netG, netI):
    x_hat = netG(z)
    x_tilde = netG(netI(x))
    return _gradient_penalty(x_hat, x_tilde, netf)

# gradient penalty for f(G(.))

def gradient_penalty_fG(x, z, netf, netG, netI):
    def _g(x):
        return netf(netG(x))
    z_hat = netI(x)
    return _gradient_penalty(z_hat, z, _g)

# gradient penalty for dual
def gradient_penalty_dual(x, z, netf, netG, netI):
    # x_hat = netG(z)
    z_hat = netI(x)
    return _gradient_penalty(z_hat, z, netf)

# Penalty for z and I(x)
def z_Qx(x, z, netf, netG, netI):
    return ((z - netI(x)).norm(p=2, dim=1)**2).mean()

# *** MMD penalty ***
# MMD loss between z and I(x)
def mmd_penalty(z_hat, z, kernel="RBF", sigma2_p=1):
    device = get_device(z)
    n = z.shape[0]
    zdim = z.shape[1]
    half_size = int((n * n - n)/2)
    #
    norms_z = z.pow(2).sum(1).unsqueeze(1)
    dots_z = torch.mm(z, z.t())
    dists_z = (norms_z + norms_z.t() - 2. * dots_z).abs()
    #
    norms_zh = z_hat.pow(2).sum(1).unsqueeze(1)
    dots_zh = torch.mm(z_hat, z_hat.t())
    dists_zh = (norms_zh + norms_zh.t() - 2. * dots_zh).abs()
    #
    dots = torch.mm(z_hat, z.t())
    dists = (norms_zh + norms_z.t() - 2. * dots).abs()
    #
    if kernel == "RBF":
        sigma2_k = torch.topk(dists.reshape(-1), half_size)[0][-1]
        sigma2_k = sigma2_k + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]
        #
        res1 = torch.exp(-dists_zh/2./sigma2_k)
        res1 = res1 + torch.exp(-dists_z/2./sigma2_k)
        res1 = torch.mul(res1, 1. - torch.eye(n).to(device))
        res1 = res1.sum() / (n*n-n)
        res2 = torch.exp(-dists/2./sigma2_k)
        res2 = res2.sum()*2./(n*n)
        stat = res1 - res2
        return stat
    #
    elif kernel == "IMQ":
        Cbase = 2 * zdim * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + dists_z) + C / (C + dists_zh)
            res1 = torch.mul(res1, 1. - torch.eye(n).to(device))
            res1 = res1.sum() / (n*n-n)
            res2 = C / (C + dists)
            res2 = res2.sum()*2./(n*n)
            stat = stat + res1 - res2
        return stat

# loss function for I update in power section

# class ExponentialMSELoss(nn.Module):
#     def __init__(self, exponent):
#         """
#         Initialize the loss function with an exponent.
        
#         Parameters:
#         exponent (float): The exponent to apply to the squared errors.
#         """
#         super(ExponentialMSELoss, self).__init__()
#         self.exponent = exponent

#     def forward(self, output, target):
#         """
#         Forward pass for the loss function.
        
#         Parameters:
#         output (torch.Tensor): The output tensor from the model (predictions).
#         target (torch.Tensor): The target tensor.
        
#         Returns:
#         torch.Tensor: The calculated loss.
#         """
#         squared_errors = (output - target) ** 2
#         exponential_errors = torch.exp(squared_errors * self.exponent) - 1
#         return exponential_errors.mean()


class ModifiedHuberLoss(nn.Module):
    def __init__(self, delta):
        """
        Initialize the Modified Huber Loss function with a delta value.
        
        Parameters:
        delta (float): The threshold at which to change from quadratic to linear loss.
        """
        super(ModifiedHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, output, target):
        """
        Forward pass for the Modified Huber Loss function.
        
        Parameters:
        output (torch.Tensor): The output tensor from the model (predictions).
        target (torch.Tensor): The target tensor.
        
        Returns:
        torch.Tensor: The calculated loss.
        """
        error1 = output - target
        error2 = output + target
        error = error1 if torch.sum(torch.abs(error1)) < torch.sum(torch.abs(error2)) else error2
        is_small_error = error < self.delta
        # error = torch.minimum(torch.abs(error1), torch.abs(error2))
        # is_small_error = error < self.delta
        quadratic = 0.5 * error**2
        linear = self.delta * (error - 0.5 * self.delta)
        return torch.where(is_small_error, quadratic, linear).mean()

# I_loss = nn.MSELoss()
# I_loss = ExponentialMSELoss(exponent=0.1)
I_loss = ModifiedHuberLoss(delta=1.0)


# *** Power Penalty ***
# power penalty for netI
# def power_penalty_D(x, eta, netI):
#     z_hat = netI(x)
#     target = eta * torch.ones(len(z_hat))
#     l2 = torch.norm(z_hat - target, p=2)
#     return l2