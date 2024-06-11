from typing import Union

import torch.nn as nn
import torch.nn.functional as F

from .projector import MLP

class SimSiam(nn.Module):
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (2048, 2048), norm_last=True, norm_embed="BN", repre_dim=None, current_step=0):
        super().__init__()
        
        self.norm_embed=norm_embed
        self.backbone_net = backbone_net

        self.repre_dim = repre_dim if repre_dim else backbone_net.fc.in_features
        self.backbone_net.fc = nn.Identity()
        self.projector = MLP(self.repre_dim, projector_hidden, bias = False, bias_last = True, norm_last=norm_last)

        self.predictor = MLP(projector_hidden[-1], (512, projector_hidden[-1]), bias = False, bias_last = True, norm_last=False)

        
        self.current_step = current_step

    def forward(self, x1, x2):

        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        if self.norm_embed:
            z1,z2 = F.normalize(z1, dim=-1, p=2), F.normalize(z2, dim=-1, p=2)
            p1,p2 = F.normalize(p1, dim=-1, p=2), F.normalize(p2, dim=-1, p=2)

        return z1.detach(),z2.detach(),p1,p2