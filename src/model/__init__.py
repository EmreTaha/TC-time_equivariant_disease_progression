from .resnet import GrayResNet
from .vicreg import VICReg, Augself_Vicreg, Essl_Vicreg, Equimod_Vicreg, TC_Vicreg, TCe_Vicreg, TC, TC_ind
from .linear_classifier import Linear_Protocoler, Traj_Protocoler
from .projector import MLP
from .byol import BYOL
from .simsiam import SimSiam

# List of all models and modules
Models = {
    "GrayResNet": GrayResNet,
    "VICReg": VICReg,
    "Linear": Linear_Protocoler,
    "Traj_Linear": Traj_Protocoler,
    "BYOL": BYOL,   
    "SimSiam": SimSiam,
    "MLP": MLP,
    "AugselfVICReg": Augself_Vicreg,
    "Essl_Vicreg": Essl_Vicreg,
    "Equimod_Vicreg": Equimod_Vicreg,
    "TC_Vicreg": TC_Vicreg,
    "TCe_Vicreg": TCe_Vicreg,
    "TC_ind": TC_ind,
    "TC": TC,
}