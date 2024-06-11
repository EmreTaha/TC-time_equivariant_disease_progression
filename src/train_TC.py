import time
import pandas as pd
import torch
import os
import numpy as np

from os import path

from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from monai.data import DataLoader

from model import Models

from utils import Barlow_augmentaions, Monai2DTransforms
from utils import OPTIMIZERS, remove_bias_and_norm_from_weight_decay
from utils import check_existing_model
from utils import args_parser
from utils import ss_data_paths
from utils import TemporalDataset, Dataset_memm_fov
from utils import TCLoss
from utils import initialize

initialize(allow_tf32=False)

args = args_parser()

def train(args):

    ### Define parameters ###
    os.makedirs(args.save_dir, exist_ok=True) 

    optim_params = {'lr': args.lr,
                    'weight_decay': args.wd}
                    
    train_params = {'num_epochs': args.epochs, 'warmup_epchs': args.warmup_epochs, 'eta_min':args.lr*1e-4}
    
    # Tensorboard
    writer = SummaryWriter(args.save_dir)

    # Save args
    with open(args.save_dir+'/args.txt', 'w') as f:
        print(args.__dict__, file=f)

    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

    ### Define model ###
    resnet = Models["GrayResNet"](args.in_ch, args.n_cl, model_type=args.backbone) #Encoding backbone
    repre_dim = resnet.model.fc.in_features

    # Equivariance model TC
    model = Models['TC'](resnet.model, projector_hidden = (args.projector_hidden,args.projector_hidden,args.projector_hidden), predictor_hidden = args.predictor_hidden,
                         norm_pred = args.norm_pred, norm_embed = False, is_3d = False, repre_dim=repre_dim, stop_gr=args.sg, pred_bias_last=args.pred_bias_last, pred_bias=args.pred_bias).to(device)
    model.cuda()
    
    ### Create datasets and dataloaders ###
    NORM = [[0.20253482627511976], [0.11396578943414482]] # Dataset specific normalization
    
    # For the contrastive pair augmentation
    train_transf = Barlow_augmentaions(224, temporal=True, scale=(0.4, 0.8), normalize=NORM, translation=True, sol_threshold=0.5)
    train_transf = Monai2DTransforms(train_transf)
    
    # For the linear evaluation testing
    test_transf = transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC,antialias=True),
                                  transforms.ConvertImageDtype(torch.float32),
                                  transforms.Normalize(*NORM)])
    test_transf = Monai2DTransforms(test_transf)

    ssl_train_scan_paths, _ = ss_data_paths(args.ssl_data_dir, volume=True)

    df_fov = pd.read_csv(args.data_dir+'/fovea.csv') # Fovea position for the OCTs

    # SSL dataset and dataloader
    ssl_train_scan_paths = ssl_train_scan_paths
    ssl_ds = Dataset_memm_fov(ssl_train_scan_paths, ssl_train_scan_paths,df_fov)
    ssl_temp_ds = TemporalDataset(ssl_ds, train_transf, cache_rate=1.0, min_max=(args.min_diff,args.max_diff),sorted=True)
    sampler = torch.utils.data.RandomSampler(ssl_temp_ds, replacement=True, num_samples=len(ssl_temp_ds)*25) # Oversampling because epochs are based on patients not scans
    ssl_dl = DataLoader(ssl_temp_ds, batch_size=args.batch_size, sampler=sampler, persistent_workers=False,
                                             worker_init_fn = lambda id: np.random.seed(id + int(time.time())), 
                                           num_workers=args.num_workers, drop_last=True, pin_memory=False)


    ### Define optimizer and scheduler ###
    parameters = remove_bias_and_norm_from_weight_decay([{"params":model.parameters()}]) if args.exclude_nb else model.parameters()
    optimizer_f = OPTIMIZERS[args.optim]
    optimizer = optimizer_f(parameters, **optim_params)
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda it : (it+1)/(train_params['warmup_epchs']*len(ssl_dl))) # Linear warmup scheduler
    
    ### Define loss ###
    criterion = TCLoss(args.vic_lambda,args.vic_mu,args.vic_nu,t=args.tc_eq,tr=args.tc_t,curve=args.tc_c,equiloss=args.equi_loss)

    ### Continue training from a saved model ###
    loss_hist = []
    lr_hist = []

    # Check if there is a saved model
    epoch_start, saved_data = check_existing_model(args.save_dir, device)

    if saved_data:
        # Extract saved model data
        msg = model.load_state_dict(saved_data['model'], strict=True)
        assert set(msg.missing_keys) == set()

        optimizer.load_state_dict(saved_data['optim'])

        if epoch_start >= train_params['warmup_epchs']:
            iters_left = iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_dl)
            curr_iter = (epoch_start-train_params['warmup_epchs'])*len(ssl_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iters_left,
                                                        eta_min=train_params['eta_min'],
                                                        last_epoch=curr_iter)
        loss_hist = saved_data['loss_hist']
        lr_hist = saved_data['lr_hist']

    ### Run Training ###
    for epoch in range(epoch_start, train_params['num_epochs']):
        epoch_loss = 0
        sim_inv_loss = 0
        std_inv_loss = 0
        cov_inv_loss = 0
        sim_equi_loss = 0
        traj_loss = 0
        model.train()
        start_time = time.time()
        for inp_dict in ssl_dl:
            optimizer.zero_grad()

            x1 = inp_dict["image"] # First timepoint
            x2 = inp_dict["image_1"] # Second timepoint
            time_diff = inp_dict["label"] # Normalized time difference
            x1,x2,time_diff = x1.to(device), x2.to(device), time_diff.to(device)

            time_diff = time_diff.reshape(-1,1)
            
            # Forward pass
            z1, z2, r2, r_equi_pred, traj = model(x1,x2,time_diff)
            loss, ind_loss = criterion(z1, z2, r2, r_equi_pred, traj) 

            # Scale Gradients
            loss.backward()

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            if bool(args.grad_norm_clip):
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_norm_clip, error_if_nonfinite=False)

            # Update Optimizer
            optimizer.step()
            
            # Scheduler every iteration for cosine deday
            scheduler.step()

            # Save loss and LR
            epoch_loss += loss.detach()
            sim_inv_loss += ind_loss[0].detach()
            std_inv_loss += ind_loss[1].detach()
            cov_inv_loss += ind_loss[2].detach()
            sim_equi_loss += ind_loss[3].detach()
            traj_loss += ind_loss[4].detach()       
            lr_hist.extend(scheduler.get_last_lr())

        epoch_time = time.time() - start_time

        # Switch to Cosine Decay after warmup period
        if epoch+1==train_params['warmup_epchs']:
            iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                        iters_left,
                                                        eta_min=train_params['eta_min'])
        
        # Log
        loss_hist.append([epoch_loss/len(ssl_dl),sim_inv_loss/len(ssl_dl),std_inv_loss/len(ssl_dl),cov_inv_loss/len(ssl_dl),sim_equi_loss/len(ssl_dl),traj_loss/len(ssl_dl)])
        print(f'Epoch: {epoch}, Loss: {epoch_loss/len(ssl_dl)}, invariance loss: {sim_inv_loss/len(ssl_dl)}, variance loss: {std_inv_loss/len(ssl_dl)}, covariance loss: {cov_inv_loss/len(ssl_dl)}, equivariance loss: {sim_equi_loss/len(ssl_dl)}, trajectory loss: {traj_loss/len(ssl_dl)}, Time epoch: {epoch_time}')
         
        writer.add_scalar("Loss/invariance", sim_inv_loss/len(ssl_dl), epoch)
        writer.add_scalar("Loss/variance", std_inv_loss/len(ssl_dl), epoch)
        writer.add_scalar("Loss/covariance", cov_inv_loss/len(ssl_dl), epoch)
        writer.add_scalar("Loss/equivariance", sim_equi_loss/len(ssl_dl), epoch)
        writer.add_scalar("Loss/trajectory", traj_loss/len(ssl_dl), epoch)

        # Save the model
        if (epoch+1)%(20*(5))==0:

            torch.save({'model':model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                        'loss_hist': loss_hist,
                        'lr_hist': lr_hist}, 
                    path.join(args.save_dir, f'epoch_{epoch+1:03}.tar'))
    
    writer.flush()

def main():
    train(args)

if __name__ == "__main__":
    main()