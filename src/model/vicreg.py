from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import MLP

class VICReg(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning (VICReg)
    
    Args:
        backbone_net (nn.Module): The backbone network used for feature extraction.
        projector_hidden (Union[int, tuple]): The hidden layer sizes of the MLP projector. Can be an integer or a tuple of integers.
        norm_last (bool): Whether to apply normalization to the last layer of the projector.
        norm_embed (bool): Whether to apply normalization to the projected embeddings.
        relu_embed (bool): Whether to apply ReLU activation to the projected embeddings.
        is_3d (bool): Whether the input data is 3D.
        repre_dim (int): The dimension of the representation. If not provided, it is inferred from the backbone network.
        proj_bias (bool): Whether to include bias in the hidden dimensions of the projector.
        proj_bias_last (bool): Whether to include bias in the last layer of the projector.
        """
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last=False, norm_embed=False, relu_embed=False, is_3d=False, repre_dim=None, proj_bias=True, proj_bias_last=False):
        super().__init__()
        
        self.norm_embed=norm_embed
        self.norm_last=norm_last
        self.relu_embed=relu_embed
        self.backbone_net = backbone_net
        self.repre_dim = repre_dim if repre_dim else self.backbone_net.fc.in_features
        if is_3d:
            self.backbone_net.blocks[5] = nn.Sequential(*[nn.AdaptiveAvgPool3d(1), nn.Flatten()])
        else:
            self.backbone_net.fc = nn.Identity()
        
        if norm_last: proj_bias_last=proj_bias
        self.projector = MLP(self.repre_dim, projector_hidden, bias = proj_bias, bias_last = proj_bias_last, norm_last=norm_last)

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))

        if self.norm_embed:
            z1,z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

        if self.relu_embed:
            z1,z2 = F.relu(z1), F.relu(z2)

        return z1,z2
    
    # Dont forget to call eval in the code!
    @torch.no_grad()
    def get_representations(self, x):
        return self.backbone_net(x)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight,mean=0.0, std=0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
class Essl_Vicreg(VICReg):
    """
    Equivariant Contrastive Learnng (ESSL)

    Args:
        backbone_net (nn.Module): The backbone network used for feature extraction.
        projector_hidden (Union[int, tuple]): The hidden layer sizes of the projector MLP. Can be an integer or a tuple of integers.
        norm_last (bool): Whether to apply normalization to the last layer of the projector MLP.
        norm_embed (bool): Whether to apply normalization to the embeddings.
        norm_func (nn.Module): The normalization function to use.

    Attributes:
        tempPredictor (MLP): Module for predicting the time difference between two images.
    """

    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last=False, norm_embed=False, norm_func=nn.BatchNorm1d, **kwargs):
        super(Essl_Vicreg,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed, **kwargs)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        self.tempPredictor = MLP(self.repre_dim*2, (self.repre_dim*2//2,self.repre_dim*2//4,1), bias = True, bias_last=True, norm_func=norm_func)
    
    def forward(self, x1, x2, x1_contr, x2_contr):
        """
        Args:
            x1 (torch.Tensor): first time step.
            x2 (torch.Tensor): second time step.
            x1_contr (torch.Tensor): Augmented contrastive view for the first time step used for contrastive part.
            x2_contr (torch.Tensor): Augmented contrastive view for the second time step used for contrastive part.
        """
        # Time pred part, based on projection
        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)
        
        # Concatenate the representations
        z_cat = torch.cat((b1, b2), dim=1)

        # Predict the time difference
        pred = self.tempPredictor(z_cat)

        # Contrastive part for time & transformation invariance in the projection space
        b3 = self.backbone_net(x1_contr)
        b4 = self.backbone_net(x2_contr)
        z1 = self.projector(b3)
        z2 = self.projector(b4)

        return z1, z2, pred
    
    # Dont forget to call eval in the code!
    @torch.no_grad()
    def get_projections(self, x):
        return self.projector(self.backbone_net(x))

class Augself_Vicreg(VICReg):
    """
    Improving Transferability of Representations via Augmentation-Aware Self-Supervision (AugSelf)

    Args:
        backbone_net (nn.Module): The backbone network used for feature extraction.
        projector_hidden (Union[int, tuple]): The hidden layer sizes of the projector MLP. Can be an integer or a tuple of integers.
        norm_last (bool): Whether to apply normalization to the last layer of the projector MLP.
        norm_embed (bool): Whether to apply normalization to the embeddings.
        norm_func (nn.Module): The normalization function to use.

    Attributes:
        tempPredictor (MLP): Module for predicting the time difference between two images.
    """

    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last='affine_BN', norm_embed=False, norm_func=nn.BatchNorm1d, **kwargs):
        super(Augself_Vicreg,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed, **kwargs)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        self.tempPredictor = MLP(self.repre_dim*2, (self.repre_dim*2//2,self.repre_dim*2//2,1), bias = False, bias_last=True, norm_func=norm_func) # This follows the original implementation
    
    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): first time step.
            x2 (torch.Tensor): second time step.

        Returns:
          the projections of x1 and x2, and the time prediction.
        """
        b1 = self.backbone_net(x1)
        z1 = self.projector(b1)

        b2 = self.backbone_net(x2)
        z2 = self.projector(b2)

        # Time prediction on representations
        z_cat = torch.cat((b1, b2),dim=1)
        pred = self.tempPredictor(z_cat)

        return z1, z2, pred

    # Dont forget to call eval in the code!
    @torch.no_grad()
    def get_projections(self, x):
        return self.projector(self.backbone_net(x))
    
class Equimod_Vicreg(VICReg):
    """
    EquiMod: An Equivariance Module to Improve Self-Supervised Learning (EquiMod)
    
    Args:
        backbone_net (nn.Module): The backbone network used for feature extraction.
        projector_hidden (Union[int, tuple]): The hidden dimensions of the invariance projector. Can be an integer or a tuple of integers.
        norm_last (str): The normalization method applied to the last layer of the invariance projector. Default is 'BN' (Batch Normalization).
        norm_embed (bool): Whether to apply normalization to the embeddings. Default is False.
        norm_func (nn.Module): The normalization function used. Default is nn.BatchNorm1d.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last='BN', norm_embed=False, norm_func=nn.BatchNorm1d,**kwargs):
        super(Equimod_Vicreg,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed, **kwargs)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        # Invariance projector comes from vicreg, so we need to define an extra equivariance projector
        self.tempProj = MLP(self.repre_dim, projector_hidden, bias = True, norm_func=norm_func, norm_last=norm_last)

        # Equivariance predictor
        self.tempPredictor = MLP(projector_hidden[-1]+1, projector_hidden[-1],bias=True,bias_last=True) 
    
    def forward(self, x1, x2, time_diff):
        """        
        Args:
            x1 (torch.Tensor): first time step.
            x2 (torch.Tensor): second time step.
            time_diff (torch.Tensor): Time difference between the two visits.
        
        Returns:
            z1_inv (torch.Tensor): Invariant representation of x1.
            z2_inv (torch.Tensor): Invariant representation of x2.
            z2_equi (torch.Tensor): Equivariant representation of x2.
            z_equi_pred (torch.Tensor): Prediction of z2_equi from z1_equi and time difference.
        """
        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)

        z1_inv = self.projector(b1)
        z2_inv = self.projector(b2)
        
        z1_equi = self.tempProj(b1)
        z2_equi = self.tempProj(b2)

        # z1_equi should be earlier in terms of time
        z2_equi_pred = self.tempPredictor(torch.concat((z1_equi,time_diff),dim=1)) # Prediction of z2_equi from z1_equi and time difference

        return z1_inv, z2_inv, z1_equi, z2_equi, z2_equi_pred
    
    @torch.no_grad()
    def get_projections(self, x):
        """
        Get the projections of the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Concatenation of the projections from the invariance projector and the equivariance projector.
        """
        return torch.cat((self.projector(self.backbone_net(x)), self.tempProj(self.backbone_net(x))), dim=1)
    
    @torch.no_grad()
    def forward_time(self, x, delta_t):
        """
        Forward pass of the Equimod_Vicreg model for time prediction.
        
        Args:
            x (torch.Tensor): Input tensor.
            delta_t (torch.Tensor): Time difference.
        
        Returns:
            torch.Tensor: Prediction of z2_equi from z1_equi and time difference.
        """
        return self.tempPredictor(torch.cat((self.tempProj(self.backbone_net(x)), delta_t),dim=1))
    
class TC_Vicreg(VICReg):
    # NOTE ablation
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), predictor_hidden=4096, norm_pred=False, norm_last='', norm_embed=False, norm_func=nn.BatchNorm1d, pred_bias=True, pred_bias_last=False, stop_gr=False, **kwargs):
        super(TC_Vicreg,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed, **kwargs)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        # Equivariance predictor
        #n_temp = True if self.norm_last=='BN' else False #this is temp, this diesnt make sene, it was used in norml_embed
        self.tempPredictor = MLP(self.repre_dim+1, (predictor_hidden,self.repre_dim), norm_embed=False, norm_last=norm_pred, bias=pred_bias, norm_func=norm_func, bias_last=pred_bias_last) #TODO make predictor part variable
        self.stop_gr = stop_gr
    def forward(self, x1, x2, time_diff):

        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)

        # b1_equi should be earlier in terms of time
        b_equi_pred = self.tempPredictor(torch.concat((b1,time_diff),dim=1)) # Prediction of z2_equi from z1_equi and time difference
        
        # Representations are normed by a BN. so it is better to l2 norm directions
        if self.norm_embed:
            n_ = torch.norm(b_equi_pred, dim=1)
            b_equi_pred = b_equi_pred / torch.reshape(n_, (-1, 1))

        z1 = self.projector(b1)
        z2 = self.projector(b2)

        if self.stop_gr:
            #TODO this is ugly
            b2_f = b2.clone().detach()
            return z1, z2, b2_f, b_equi_pred

        return z1, z2, b2, b_equi_pred
    
    # Dont forget to call eval in the code!
    @torch.no_grad()
    def get_representations(self, x):
        return self.backbone_net(x)
    
    @torch.no_grad()
    def get_projections(self, x):
        #TODO fix this later
        return torch.cat((self.projector(self.backbone_net(x)), self.tempProj(self.backbone_net(x))), dim=1)
    
    @torch.no_grad()
    def forward_time(self, x, delta_t):
        b1 = self.backbone_net(x)

        b_next = self.tempPredictor(torch.concat((b1,delta_t),dim=1)) # Prediction of z2_equi from z1_equi and time difference
        
        # Representations are normed by a BN. so it is better to l2 norm directions
        if self.norm_embed:
            n_ = torch.norm(b_next, dim=1)
            b_next = b_next / torch.reshape(n_, (-1, 1))
        return b_next

    @torch.no_grad()
    def forward_repr(self, representation, delta_t):
        b_next = self.tempPredictor(torch.concat((representation,delta_t),dim=1)) # Prediction of z2_equi from z1_equi and time difference
        
        # Representations are normed by a BN. so it is better to l2 norm directions
        if self.norm_embed:
            n_ = torch.norm(b_next, dim=1)
            b_next = b_next / torch.reshape(n_, (-1, 1))
        return b_next
    
class TCe_Vicreg(VICReg):
    # NOTE ablation
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), norm_last='BN', norm_embed=False, norm_func=nn.BatchNorm1d, **kwargs):
        super(TCe_Vicreg,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed, **kwargs)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        # Equivariance predictor
        #n_temp = True if self.norm_last=='BN' else False #TODO this is temp
        self.tempPredictor = MLP(self.repre_dim+1, (projector_hidden[-1],self.repre_dim), norm_embed=False, norm_last=False, bias=False) #TODO make predictor part variable
    
    def forward(self, x1, x2, time_diff):

        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)

        # b1_equi should be earlier in terms of time
        b_equi_pred = self.tempPredictor(torch.concat((b1,time_diff),dim=1)) # Prediction of trajectory from z1_equi and time difference
        
        # Representations are normed by a BN. so it is better to l2 norm directions
        if self.norm_embed:
            n_ = torch.norm(b_equi_pred, dim=1)
            b_equi_pred = b_equi_pred / torch.reshape(n_, (-1, 1))

        b_equi_pred = b1 + b_equi_pred # Prediction of z2_equi from z1_equi and time difference        

        z1 = self.projector(b1)
        z2 = self.projector(b2)

        return z1, z2, b2, b_equi_pred
    
    # Dont forget to call eval in the code!
    @torch.no_grad()
    def get_representations(self, x):
        return self.backbone_net(x)
    
    @torch.no_grad()
    def get_projections(self, x):
        #TODO fix this later
        return torch.cat((self.projector(self.backbone_net(x)), self.tempProj(self.backbone_net(x))), dim=1)
    
    @torch.no_grad()
    def get_trajectory(self, b1, delta_t):
        traj = self.tempPredictor(torch.concat((b1,delta_t),dim=1)) # Prediction of z2_equi from z1_equi and time difference
        
        # Representations are normed by a BN. so it is better to l2 norm directions
        if self.norm_embed:
            n_ = torch.norm(traj, dim=1)
            traj = traj / torch.reshape(n_, (-1, 1))

        return traj

    @torch.no_grad()
    def forward_time(self, x, delta_t):
        b1 = self.backbone_net(x)
        traj = self.get_trajectory(b1, delta_t)
        b_next = b1 + traj
        return b_next

    @torch.no_grad()
    def forward_repr(self, representation, delta_t):
        traj = self.get_trajectory(representation, delta_t)
        b_next = representation + traj
        return b_next

class TC(VICReg):
    """
    Time-Equivariant Contrastive Learning for Degenerative Disease Progression in Retinal OCT (TC)

    Args:
        backbone_net (nn.Module): Backbone network for feature extraction.
        projector_hidden (Union[int, tuple]): Number of hidden units in the projector MLP. Can be an integer or a tuple of integers.
        predictor_hidden (int): Number of hidden units in the predictor MLP.
        norm_pred (bool): Whether to apply normalization to the predictor MLP output.
        norm_last (str): Normalization type to be applied to the last layer of the predictor MLP.
        norm_embed (bool): Whether to apply normalization to the representations before MLP projector.
        norm_func (nn.Module): Normalization function to be used in MLPs.
        pred_bias (bool): Whether to include bias in the temporal predictor MLP.
        pred_bias_last (bool): Whether to include bias in the last layer of the temporal predictor MLP.
        stop_gr (bool): Whether to stop gradient propagation of the future branch in the Siamese network.
        **kwargs: Additional keyword arguments.

    Attributes:
        tempPredictor (nn.Module): Equivariance predictor MLP.
        stop_gr (bool): Whether to stop gradient propagation during training.

    """

    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), predictor_hidden=4096, norm_pred=False, norm_last='', norm_embed=False, norm_func=nn.BatchNorm1d, pred_bias=True, pred_bias_last=False, stop_gr=False, **kwargs):
        super(TC,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed, **kwargs)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        # Equivariance predictor
        self.tempPredictor = MLP(self.repre_dim+1, (predictor_hidden,self.repre_dim), norm_embed=False, norm_last=norm_pred, bias=pred_bias, norm_func=norm_func, bias_last=pred_bias_last)
        self.stop_gr = stop_gr

        self.apply(self._init_weights)

    def forward(self, x1, x2, time_diff):
        """
        Forward pass of the TC model.

        Args:
            x1 (torch.Tensor): Input tensor for the first time point.
            x2 (torch.Tensor): Input tensor for the second time points.
            time_diff (torch.Tensor): Normalized time difference between the two visits.

        Returns:
            - z1: Projection of the first visit.
            - z2: Projection of the second visit.
            - b2: Representation of the second visit to be used as the future representation label
            - b_equi_pred: Prediction of the second visit's representation from the first visit's representation and time difference.
            - traj: Predicted displacement map (DM) from the first visit's representation and time difference.

        """
        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)

        traj = self.tempPredictor(torch.cat((b1,time_diff),dim=1)) # Prediction of trajectory from z1_equi and time difference
        
        if self.norm_embed:
            n_ = torch.norm(traj, dim=1)
            traj = traj / torch.reshape(n_, (-1, 1))

        b_equi_pred = b1 + traj

        z1 = self.projector(b1)
        z2 = self.projector(b2)

        if self.stop_gr:
            b2_f = b2.clone().detach()
            return z1, z2, b2_f, b_equi_pred, traj

        return z1, z2, b2, b_equi_pred, traj
    
    @torch.no_grad()
    def get_representations(self, x):
        """
        Get the representations from the backbone network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Representations obtained from the backbone network.

        """
        return self.backbone_net(x)
    
    @torch.no_grad()
    def get_projections(self, x):
        """
        Get the projections from the backbone network and the projector.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenation of the projections obtained from the backbone network and the projector.

        """
        return torch.cat((self.projector(self.backbone_net(x)), self.tempProj(self.backbone_net(x))), dim=1)
    
    @torch.no_grad()
    def get_trajectory(self, b1, delta_t):
        """
        Get the trajectory prediction.

        Args:
            b1 (torch.Tensor): Representation of the first frame.
            delta_t (torch.Tensor): Time difference between the two frames.

        Returns:
            torch.Tensor: Predicted trajectory.

        """
        traj = self.tempPredictor(torch.cat((b1,delta_t),dim=1))
        
        if self.norm_embed:
            n_ = torch.norm(traj, dim=1)
            traj = traj / torch.reshape(n_, (-1, 1))

        return traj

    @torch.no_grad()
    def forward_time(self, x, delta_t):
        """
        Forward pass of the TC model for a single frame and time difference.

        Args:
            x (torch.Tensor): Input tensor.
            delta_t (torch.Tensor): Time difference.

        Returns:
            torch.Tensor: Prediction of the next frame's representation.

        """
        b1 = self.backbone_net(x)
        traj = self.get_trajectory(b1, delta_t)
        b_next = b1 + traj
        return b_next

    @torch.no_grad()
    def forward_repr(self, representation, delta_t):
        """
        Forward pass of the TC model for a given representation and time difference.

        Args:
            representation (torch.Tensor): Input representation.
            delta_t (torch.Tensor): Time difference.

        Returns:
            torch.Tensor: Prediction of the next representation.

        """
        traj = self.get_trajectory(representation, delta_t)
        b_next = representation + traj
        return b_next
    
class TC_ind(VICReg):
    '''
    Ablation of the temporal predictor, the trajectory is directly predicted from the time difference, independent of the representations. Similar to how rotation matrix is independent from the image.

    Args:
        backbone_net (nn.Module): The backbone network used for feature extraction.
        projector_hidden (Union[int, tuple]): The hidden dimensions of the projector MLP. Can be an integer or a tuple of integers.
        predictor_hidden (int): The hidden dimension of the temporal predictor MLP.
        norm_pred (bool): Whether to apply normalization to the temporal predictor output.
        norm_last (str): The normalization method to be applied to the last layer of the projector MLP.
        norm_embed (bool): Whether to apply normalization to the trajectory embeddings.
        norm_func (nn.Module): The normalization function to be used.
        pred_bias (bool): Whether to include bias in the temporal predictor MLP.
        pred_bias_last (bool): Whether to include bias in the last layer of the temporal predictor MLP.
        stop_gr (bool): Whether to stop gradient propagation during forward pass.

    Attributes:
        tempPredictor (MLP): The temporal predictor MLP.
        stop_gr (bool): Whether to stop gradient propagation during forward pass.

    Methods:
        forward(x1, x2, time_diff): Performs forward pass of the TC_ind model.
        get_representations(x): Returns the representations obtained from the backbone network.
        get_projections(x): Returns the projections obtained from the backbone network and the projector MLP.
        get_trajectory(b1, delta_t): Returns the trajectory predicted from the given representation and time difference.
        forward_time(x, delta_t): Performs forward pass of the TC_ind model with a given input and time difference.
        forward_repr(representation, delta_t): Performs forward pass of the TC_ind model with a given representation and time difference.
    '''
    def __init__(self, backbone_net, projector_hidden: Union[int, tuple] = (8192,8192,8192), predictor_hidden=4096, norm_pred=False, norm_last='', norm_embed=False, norm_func=nn.BatchNorm1d, pred_bias=True, pred_bias_last=False, stop_gr=False, **kwargs):
        super(TC_ind,self).__init__(backbone_net, projector_hidden, norm_last, norm_embed, **kwargs)
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)

        # Equivariance predictor
        self.tempPredictor = MLP(1, (predictor_hidden,self.repre_dim), norm_embed=False, norm_last=norm_pred, bias=pred_bias, norm_func=norm_func, bias_last=pred_bias_last)
        self.stop_gr = stop_gr

        self.apply(self._init_weights)

    def forward(self, x1, x2, time_diff):
        '''
        Performs forward pass of the TC_ind model.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.
            time_diff (torch.Tensor): The time difference between x1 and x2.

        Returns:
            z1 (torch.Tensor): The projection of x1.
            z2 (torch.Tensor): The projection of x2.
            b2 (torch.Tensor): The backbone output of x2.
            b_equi_pred (torch.Tensor): The predicted backbone output of x2 based on x1 and time_diff.
            traj (torch.Tensor): The predicted trajectory based on x1 and time_diff.
        '''
        b1 = self.backbone_net(x1)
        b2 = self.backbone_net(x2)

        traj = self.tempPredictor(time_diff) # Prediction of trajectory from z1_equi and time difference
        
        if self.norm_embed:
            n_ = torch.norm(traj, dim=1)
            traj = traj / torch.reshape(n_, (-1, 1))

        b_equi_pred = b1 + traj # Prediction of z2_equi from z1_equi and time difference        

        z1 = self.projector(b1)
        z2 = self.projector(b2)

        if self.stop_gr:
            b2_f = b2.clone().detach()
            return z1, z2, b2_f, b_equi_pred, traj

        return z1, z2, b2, b_equi_pred, traj
    
    @torch.no_grad()
    def get_representations(self, x):
        '''
        Returns the representations obtained from the backbone network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            representations (torch.Tensor): The representations obtained from the backbone network.
        '''
        return self.backbone_net(x)
    
    @torch.no_grad()
    def get_projections(self, x):
        '''
        Returns the projections obtained from the backbone network and the projector MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            projections (torch.Tensor): The projections obtained from the backbone network and the projector MLP.
        '''
        return torch.cat((self.projector(self.backbone_net(x)), self.tempProj(self.backbone_net(x))), dim=1)
    
    @torch.no_grad()
    def get_trajectory(self, b1, delta_t):
        '''
        Returns the trajectory predicted from the given representation and time difference.

        Args:
            b1 (torch.Tensor): The input representation.
            delta_t (torch.Tensor): The time difference.

        Returns:
            traj (torch.Tensor): The predicted trajectory.
        '''
        traj = self.tempPredictor(torch.concat((b1,delta_t),dim=1))
        
        if self.norm_embed:
            n_ = torch.norm(traj, dim=1)
            traj = traj / torch.reshape(n_, (-1, 1))

        return traj

    @torch.no_grad()
    def forward_time(self, x, delta_t):
        '''
        Performs forward pass of the TC_ind model with a given input and time difference.

        Args:
            x (torch.Tensor): The input tensor.
            delta_t (torch.Tensor): The time difference.

        Returns:
            b_next (torch.Tensor): The predicted backbone output of the next time step.
        '''
        b1 = self.backbone_net(x)
        traj = self.get_trajectory(b1, delta_t)
        b_next = b1 + traj
        return b_next

    @torch.no_grad()
    def forward_repr(self, representation, delta_t):
        '''
        Performs forward pass of the TC_ind model with a given representation and time difference.

        Args:
            representation (torch.Tensor): The input representation.
            delta_t (torch.Tensor): The time difference.

        Returns:
            b_next (torch.Tensor): The predicted backbone output of the next time step.
        '''
        traj = self.get_trajectory(representation, delta_t)
        b_next = representation + traj
        return b_next