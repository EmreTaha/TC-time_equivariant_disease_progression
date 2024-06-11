from typing import Union

import torch.nn as nn

from .norms import IterNorm

class L2Norm(nn.Module):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim)

class MLP(nn.Module):
    # Projector/Expander network for contrastive/non-contrastive embeddings
    def __init__(self, in_dim: int,
                 hidden_dims: Union[int, tuple],
                 bias = True, norm_last = False, bias_last = False,
                 norm_embed=False, norm_func = nn.BatchNorm1d, actv_func = nn.ReLU(inplace=True)):
        super().__init__()
        
        if hidden_dims == 0:
            self.mlp = nn.Sequential(nn.Identity())
            return

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        
        hidden_dims = (in_dim,) + hidden_dims

        mlp = []

        if norm_embed:
            mlp.append(nn.BatchNorm1d(hidden_dims[0], affine=False))

        for i in range(len(hidden_dims) - 2):
            mlp.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1], bias = bias),
                        norm_func(hidden_dims[i+1]),
                        actv_func])
            
        if norm_last=="SN":
            ''' Spectral Normalization follows different than the rest of the norms, so this split is necessary'''
            mlp.append(nn.utils.spectral_norm(nn.Linear(hidden_dims[-2], hidden_dims[-1], bias = bias_last)))
        else:
            mlp.extend([nn.Linear(hidden_dims[-2], hidden_dims[-1], bias = bias_last)])

        if norm_last=='BN':
            mlp.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))
        elif norm_last=='affine_BN':
            mlp.append(nn.BatchNorm1d(hidden_dims[-1], affine=True))
        elif norm_last=='IN':
            mlp.append(IterNorm(hidden_dims[-1], num_groups=64, T=5, dim=2))
        elif norm_last=='L2':
            mlp.append(L2Norm())
        
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)