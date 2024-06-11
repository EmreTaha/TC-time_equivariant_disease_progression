import random
import numpy as np
import torch

def initialize(seed=3407, allow_tf32=False, deterministic=True):
    torch.cuda.empty_cache()

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    np.random.seed(seed)

def worker_init_fn(worker_id):                                                          
    np.random.seed(torch.initial_seed() // 2**32 + worker_id)
    random.seed(torch.initial_seed() // 2**32 + worker_id)