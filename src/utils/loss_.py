import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import os

class VicLoss(nn.modules.loss._Loss):
    """
    Computes the VICReg loss.

    Args:
        λ (float): Weight for the invariance loss. Default is 25.
        μ (float): Weight for the variance loss. Default is 25.
        ν (float): Weight for the covariance loss. Default is 1.
        γ (float): Threshold for the variance loss. Default is 1.
        ϵ (float): Small value added to the variance to avoid division by zero. Default is 1e-4.
        is_distributed (bool): Flag indicating if the loss is used in a distributed setting. Default is False.
    """

    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, is_distributed: bool = False):
        super(VicLoss,self).__init__()
        self.lambd = λ
        self.mu = μ
        self.nu = ν
        self.gamma = γ
        self.eps = ϵ
        self.is_distributed = is_distributed

    def _off_diagonal(self, x):
        """
        Returns a flattened view of the off-diagonal elements of a square matrix.

        Args:
            x (torch.Tensor): Input square matrix.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ):
        """
        Args:
            z1 (torch.Tensor): Input tensor of shape (N, D).
            z2 (torch.Tensor): Input tensor of shape (N, D).
            λ (float): Weight for the invariance loss.
            μ (float): Weight for the variance loss.
            ν (float): Weight for the covariance loss.
            γ (float): Threshold for the variance loss.
            ϵ (float): Small value added to the variance to avoid division by zero.

        Returns:
            torch.Tensor: Final loss value.
            tuple: Tuple containing the individual losses (sim_loss, std_loss, cov_loss) for logging.
        """
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        sim_loss = F.mse_loss(z1, z2)

        if self.is_distributed:
            z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
            z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)

        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() / 2  + torch.relu(γ - std_z2).mean() / 2

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # covariance loss
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / (D) #(2 * D)
        
        sim_lossd,std_lossd,cov_lossd =sim_loss.clone().detach(),std_loss.clone().detach(),cov_loss.clone().detach()

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_lossd, std_lossd, cov_lossd)

    def forward(self,z1,z2):
        return self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps)

class TemporalVicLoss(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1.):
        super(TemporalVicLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t

    def timeloss(self,t1,t2,diff):
        return self.t*F.mse_loss(abs(t1-t2), diff)

    def forward(self,z1,z2,t1,t2,diff):
        if self.lambd or self.mu or self.nu:
            cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        else:
            cssl_loss, ind_loss = 0, (0,)
        time_loss = self.timeloss(t1,t2,diff)

        ind_loss = ind_loss + (time_loss,)
        return  cssl_loss+time_loss, ind_loss

class TemporalVicLossv2(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1.):
        super(TemporalVicLossv2,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t

    def timeloss(self,t1,diff):
        return self.t*F.mse_loss(t1, diff)

    def forward(self,z1,z2,t1,diff):
        if self.lambd or self.mu or self.nu:
            cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        else:
            cssl_loss, ind_loss = 0, (0,)
        time_loss = self.timeloss(t1,diff)

        time_lossd = time_loss.clone().detach()
        ind_loss = ind_loss + (time_lossd,)
        loss = cssl_loss+time_loss
        return  loss, ind_loss

    
class EquimodLoss(VicLoss):
    ### Following https://github.com/facebookresearch/SIE
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4):
        super(EquimodLoss,self).__init__(λ, μ, ν, γ, ϵ)

    def forward(self,z1_inv,z2_inv,z1_equi,z2_equi,z_equi_pred):
        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.lambd, self.mu, self.nu, self.gamma, self.eps)
        
        # Invariance weight is 0 because the term is calculated between predicted z2 and the real z2
        equi_loss, ind_equi_loss = self.loss_fn(z1_equi, z2_equi, 0, self.mu, self.nu, self.gamma, self.eps)
        
        # Invariance for the equivariant part
        equi_sim_loss = F.mse_loss(z_equi_pred, z2_equi)
        equi_loss = equi_loss + self.lambd*equi_sim_loss

        total_loss = inv_loss + 0.45*equi_loss

        ind_loss = ind_inv_loss + (equi_sim_loss,) + ind_equi_loss[1:]
        return  total_loss, ind_loss
    
class TCLoss_abl(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1e-2):
        super(TCLoss_abl,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t

    def forward(self,z1_inv,z2_inv,r2, r_equi_pred):
        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.lambd, self.mu, self.nu, self.gamma, self.eps) 
        equi_loss = F.mse_loss(r_equi_pred, r2)

        total_loss = inv_loss + self.lambd*self.t*equi_loss

        equi_lossd = equi_loss.clone().detach()
        ind_loss = ind_inv_loss + (equi_lossd,)

        return  total_loss, ind_loss

class TCLoss(VicLoss):
    """
    Custom loss function for TC model.

    Args:
        λ (float): Weight for the inverse loss term. Default is 25.0.
        μ (float): Weight for the inverse loss term. Default is 25.0.
        ν (float): Weight for the inverse loss term. Default is 1.0.
        γ (float): Weight for the inverse loss term. Default is 1.0.
        ϵ (float): Small value for numerical stability. Default is 1e-4.
        t (float): Weight for the equilibrium loss term. Default is 1e-2.
        tr (float): Weight for the trajectory loss term. Default is 1e-2.
        curve (float): Curve parameter for the trajectory loss term. Default is 1.0.
        equiloss (str): Type of loss function for the equilibrium loss term. Can be 'mse' or 'cos'. Default is 'mse'.

    Attributes:
        t (float): Weight for the equilibrium loss term.
        curve (float): Curve parameter for the trajectory loss term.
        tr (float): Weight for the trajectory loss term.
        equiloss (str): Type of loss function for the equilibrium loss term.

    Methods:
        forward(z1_inv, z2_inv, r2, r_equi_pred, trajectory):
            Computes the total loss and individual loss terms.

    """

    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1e-2, tr: float = 1e-2, curve: float = 1.0, equiloss: str = 'mse'):
        super(TCLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t
        self.curve = curve
        self.tr = tr
        self.equiloss = equiloss

    def forward(self,z1_inv,z2_inv, r2, r_equi_pred, trajectory):
        """
        Computes the total loss and individual loss terms.

        Args:
            z1_inv (tensor): Inverse of z1.
            z2_inv (tensor): Inverse of z2.
            r2 (tensor): Target representation r2.
            r_equi_pred (tensor): Predicted representation for equilibrium.
            trajectory (tensor): Trajectory tensor.

        Returns:
            total_loss (tensor): Total loss.
            ind_loss (tuple): Tuple of individual loss terms.

        """
        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.lambd, self.mu, self.nu, self.gamma, self.eps)
        
        if self.equiloss == 'mse':
            equi_loss = F.mse_loss(r_equi_pred, r2)
        elif self.equiloss == 'cos':
            equi_loss = 1 - F.cosine_similarity(r_equi_pred, r2, dim=1).mean()
        
        norm = torch.norm(trajectory, dim=1)

        traj_loss = torch.log1p((norm.neg() * self.curve).exp()).mean() # More numerically stable

        total_loss = inv_loss + self.lambd*self.t*equi_loss + self.lambd*self.tr*traj_loss

        equi_lossd, traj_lossd = equi_loss.clone().detach(), traj_loss.clone().detach()
        ind_loss = ind_inv_loss + (equi_lossd,) + (traj_lossd,)
        
        return  total_loss, ind_loss
    
class TINC(VicLoss):
    """
    TINC (2206.15282) loss.

    Args:
        λ (float): Weight for the similarity loss term. Default is 25.0.
        μ (float): Weight for the variance loss term. Default is 25.0.
        ν (float): Weight for the covariance loss term. Default is 1.0.
        γ (float): Threshold for the standard deviation loss term. Default is 1.0.
        ϵ (float): Small value added to the variance to avoid division by zero. Default is 1e-4.

    Inherits from the VicLoss class.
    """
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, insensitive = "one"):
        super(TINC,self).__init__(λ, μ, ν, γ, ϵ)
        self.insensitive = insensitive

    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ, time_label):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        margin = F.mse_loss(z1, z2, reduction='none').mean(axis=1)-time_label #mean is the across the representation dimension

        if self.insensitive == "two": margin = margin**2
        sim_loss = F.relu(margin).mean() # Reduction needs to be done because margin is specific to each example
        
        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() / 2  + torch.relu(γ - std_z2).mean() / 2

        z1 = z1 - z1.mean(dim=0) #Demeaning is added later!
        z2 = z2 - z2.mean(dim=0) #Demeaning is added later!

        # covariance loss
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / (D) #(2 * D)

        sim_lossd,std_lossd,cov_lossd =sim_loss.clone().detach(),std_loss.clone().detach(),cov_loss.clone().detach()

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_lossd, std_lossd, cov_lossd)
    
    def forward(self,z1,z2,time_label):
        """
        Args:
            z1 (torch.Tensor): Tensor of shape (N, D) representing the first visit representations.
            z2 (torch.Tensor): Tensor of shape (N, D) representing the second visit representations.
            time_label (torch.Tensor): Tensor of shape (N,) representing the time difference.
        """
        cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps, time_label) 

        return  cssl_loss, ind_loss      

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass