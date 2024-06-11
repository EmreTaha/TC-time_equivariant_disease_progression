import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from utils import create_cacheds_dl, Dataset_memm_fov
from utils import Monai2DTransforms, bscan_sup_transformsv3
from utils import initialize
from utils import args_parser

from model import Models

import time

args = args_parser()

def train(args,return_model=False):
    initialize()
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    
    # Define the model and load pretrained weights
    model  = Models["GrayResNet"](args.in_ch, args.n_cl, model_type=args.backbone)

    # load pretrained weights
    if args.pretrained:
        if args.pretrained_model == "imagenet": # Use imagenet weights except the first convolutional layer because rgb vs grayscale
            model  = Models["GrayResNet"](args.in_ch, args.n_cl, model_type=args.backbone, weights="ResNet50_Weights.IMAGENET1K_V1")#2")
        else: # Else use saved weights
            saves = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
            saves = {"model." + key[key.find('.')+1:] : val for key, val in saves['model'].items() if 'backbone_net' in key.lower()}
            msg = model.load_state_dict(saves, strict=False)

    repre_dim = 2048
    model.model.fc =  nn.Identity()

    model.cuda()
    
    ### Create datasets and dataloaders ###
    df_train = pd.read_csv(args.data_dir+'/train.csv')
    df_test = pd.read_csv(args.data_dir+'/test.csv')
    df_val = pd.read_csv(args.data_dir+'/val.csv')
    df_fov = pd.read_csv(args.data_dir+'/splits/fovea.csv')

    train_paths = df_train['Filepath'].tolist()
    train_labels = df_train['label'].tolist()
    train_scan = list(set(list(zip(train_paths,train_labels))))

    test_paths = df_test['Filepath'].tolist()
    test_labels = df_test['label'].tolist()
    test_scan = list(set(list(zip(test_paths,test_labels))))

    val_paths = df_val['Filepath'].tolist()
    val_labels = df_val['label'].tolist()
    val_scan = list(set(list(zip(val_paths,val_labels))))

    # End

    NORM = [[0.20253482627511976], [0.11396578943414482]]

    train_transform = bscan_sup_transformsv3(NORM, rotation=5)
    train_transform = Monai2DTransforms(train_transform)

    test_transform = transforms.Compose([transforms.Resize((224,224),InterpolationMode.BICUBIC,antialias=True),
                                        transforms.ConvertImageDtype(torch.float32),
                                        transforms.Normalize(*NORM)])
    test_transform = Monai2DTransforms(test_transform)

    train_scan_paths,train_scan_labels = map(list, zip(*train_scan))
    ds_train_scan = Dataset_memm_fov(train_scan_paths,train_scan_labels,df_fov)
    
    val_scan_paths,val_scan_labels = map(list, zip(*val_scan))
    ds_val_scan = Dataset_memm_fov(val_scan_paths,val_scan_labels,df_fov)

    test_scan_paths,test_scan_labels = map(list, zip(*test_scan))
    ds_test_scan = Dataset_memm_fov(test_scan_paths,test_scan_labels,df_fov)

    trainloader = create_cacheds_dl(ds_train_scan, train_transform, cache_rate=1.0, batch_size=args.batch_size,
                                          shuffle=True, num_workers=12, drop_last=True,
                                         worker_fn = lambda id: np.random.seed(id + int(time.time())), progress=False)
    testloader = create_cacheds_dl(ds_test_scan, test_transform, shuffle=False, num_workers=12, batch_size=4,  cache_rate=1.0, progress=False)
    valloader = create_cacheds_dl(ds_val_scan, test_transform, shuffle=False, num_workers=12,  batch_size=4,cache_rate=1.0, progress=False)

    ### Define Linear Protocolor ###
    linear_proto = Models['Linear'](model, num_classes=1, out_dim=repre_dim, device=device)

    # Train the linear protocol
    linear_proto.train(trainloader, args.epochs, lr = args.lr, class_weights=args.lw, dictionary=True, amp=args.scale)

    # Test the linear protocol, valloader is used for probability thresholding
    test_accs, test_f1, test_prauc, test_auc, b_accs = linear_proto.get_3Dmetrics_thr(testloader, valloader, dictionary=True)
    if return_model:
        return linear_proto
    
    return (test_accs, test_f1, test_prauc, test_auc, b_accs)
    
def main():
    t0 = time.time()
    train(args)
    train_time = time.time() - t0
    print("Total runtime: ", train_time)

if __name__ == "__main__":
    main()