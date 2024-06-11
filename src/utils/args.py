import argparse

def args_parser():
    parser = argparse.ArgumentParser()    

    parser.add_argument('--data_dir', type=str,
        help='data path to the volumes')

    parser.add_argument('--ssl_data_dir', type=str,
        help='data path to the volumes')

    parser.add_argument('--optim', type=str, default="AdamW",
        help='Optimizer type')

    parser.add_argument('--ag', default=False, action='store_true', required=False,
        help='whether use amsgrad with Adam or not')

    parser.add_argument('--exclude_nb', default=False, action='store_true', required=False,
        help='Exclude norm and biases')

    parser.add_argument('--save_dir', type=str, default="/saved_models/ssl/experiment_1/",
        help='model and details save path')
    
    parser.add_argument('--epochs', type=int, default=100,
        help='max. num of epochs for training')
    
    parser.add_argument('--in_ch', type=int, default=1,
        help='number of input channels')

    parser.add_argument('--batch_size', type=int, default=16,
        help='batch size')

    parser.add_argument('--n_cl', type=int, default=1,
        help='number of classes, should be arranged accordingly with binning')

    parser.add_argument('--lr', type=float, default=5e-5,
        help='Main learning rate')

    parser.add_argument("--lr_sch", type=str,
        help='What kind of learning rate scheduler to use')

    parser.add_argument('--grad_norm_clip', type=float, default=5.0,
        help='Enable gradient norm clipping')

    parser.add_argument('--backbone', type=str, default="resnet50",
        help='Backbone model')
    
    parser.add_argument('--projector_hidden', type=int, default=4096,
        help='Hidden dimension of the projector')

    parser.add_argument('--predictor_hidden', type=int, default=4096,
        help='Hidden dimension of the equivariance predictor')
    
    parser.add_argument('--pred_bias', default=False, action='store_true', required=False,
        help='Bias inside the temporal predictor')
    
    parser.add_argument('--pred_bias_last', default=False, action='store_true', required=False,
        help='Bias for the output of the temporal predictor')

    parser.add_argument('--pretrained', default=False, action='store_true', required=False,
        help='Load pretrained backbone')

    parser.add_argument('--pretrained_model', type=str, default='',
        help='Path to the pretrained backbone weights')

    parser.add_argument('--warmup_epochs', type=int, default=10,
        help='Number of warmup epochs')

    parser.add_argument('--wd', type=float, default=0,
        help='Weight decay')

    parser.add_argument('--lw', type=float, default=5.0,
        help='Class weights of the loss')

    parser.add_argument('--norm_embed', default=False, action='store_true', required=False,
        help='L2-Normalize representation from the SSL')
    
    parser.add_argument('--norm_last', type=str, default='',
        help='Normalize projection from the SSL, options are iternorm IN, batchnorm BN, l2 norm L2, affine_BN')
    
    parser.add_argument('--norm_pred', type=str, default='',
        help='Normalize predictor from the SSL, options are iternorm IN, batchnorm BN, l2 norm L2, affine_BN, SN')

    parser.add_argument('--equi_loss', type=str, default='mse',
        help='Equivariance loss type, options are  mse, bce')
    
    parser.add_argument('--num_workers', type=int, default=6,
        help='Number of workers for the dataloaders')

    parser.add_argument('--scale', default=False, action='store_true', required=False,
        help='Uses gradscaling and AMP')

    parser.add_argument('--off_lambda', type=float, default=0.0051,
        help='Off diagonal loss term weight')
    
    parser.add_argument('--vic_lambda', type=float, default=25.,
        help='Vicreg lambda coefficient')
    
    parser.add_argument('--vic_mu', type=float, default=25.,
        help='Vicreg mu coefficient')
    
    parser.add_argument('--vic_nu', type=float, default=1.,
        help='Vicreg nu coefficient')

    parser.add_argument('--tc_t', type=float, default=0.02,
        help='Trajectory regularizer coefficient')
    
    parser.add_argument('--tc_eq', type=float, default=1e-2,
        help='Equivariance loss coeffeicient')

    parser.add_argument('--tc_c', type=float, default=1.0,
        help='Trajectory regularizer curve')
    
    parser.add_argument('--sg', default=False, action='store_true', required=False,
        help='Stop gradient in the future branch for equivariance loss')

    parser.add_argument('--min_diff', type=int, default=90,
        help='Minimum number of difference between scans in days')

    parser.add_argument('--max_diff', type=int, default=540,
        help='Maximum number of difference between scans in days')

    parser.add_argument('--beta1', type=float, default=0.9,
        help='Beta1 of adam/lion')
    
    parser.add_argument('--beta2', type=float, default=0.999,
        help='Beta2 of adam')

    args = parser.parse_args()

    return args