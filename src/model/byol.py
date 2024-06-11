from typing import Union
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import MLP

class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        """Updates momentum parameters using exponential moving average.

        Args:
            base_tau (float, optional): base value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.996.
            final_tau (float, optional): final value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 1.0.
        """

        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        """Performs the momentum update for each param group.

        Args:
            online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
            momentum_net (nn.Module): momentum network (e.g. momentum backbone,
                momentum projection, etc...).
        """

        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )


class BYOL(nn.Module):
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (4096, 256), m=0.996, norm_last=False, norm_embed=True, repre_dim=None, current_step=0):
        super().__init__()
        
        self.m = m
        self.norm_embed=norm_embed
        self.backbone_net = None

        self.repre_dim = repre_dim if repre_dim else backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        projector = MLP(self.repre_dim, projector_hidden, bias = False, norm_last=norm_last)
        self.online_encoder = nn.Sequential(OrderedDict([('backbone_net',backbone_net), ('projector',projector)]))

        self.predictor = MLP(projector_hidden[-1], projector_hidden, bias = False, norm_last=norm_last)

        self.target_net = self._get_target_encoder()
        
        self.current_step = current_step

    @torch.no_grad()
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        target_encoder.eval() # That is important
        return target_encoder
    
    @torch.no_grad()
    def _update_target_network(self):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_net.parameters()):
            #param_k.data = param_k.data.mul_(self.m).add_(1. - self.m, param_q.data)
            param_k.data = param_k.data * self.m + (1. - self.m) * param_q.data

    @torch.no_grad()
    def update_tau(self, max_steps):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing."""
        # Checked manually, it works
        self.m = (1.0 - (1.0 - self.m) * (math.cos(math.pi * self.current_step / max_steps) + 1) / 2 )
        self.current_step += 1

    def forward(self, x1, x2):

        z1 = self.predictor(self.online_encoder(x1))
        z2 = self.predictor(self.online_encoder(x2))

        with torch.no_grad():
            p1 = self.target_net(x1).detach()
            p2 = self.target_net(x2).detach()

        if self.norm_embed:
            z1,z2 = F.normalize(z1, dim=-1, p=2), F.normalize(z2, dim=-1, p=2)
            p1,p2 = F.normalize(p1, dim=-1, p=2), F.normalize(p2, dim=-1, p=2)

        return z1,z2,p1,p2
    
    @torch.no_grad()
    def _training_completed(self):
        # Just to make model names competible with other models
        self.backbone_net = copy.deepcopy(self.target_net.backbone_net)
        self.backbone_net.fc = nn.Identity()
        del self.target_net
        self.target_net = None