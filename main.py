# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import argparse
import warnings
from Model import SDAGCN
from Experiment.Experiment_merge import Exp_merge
from Experiment.Experiment_PINN import Exp_PINN

warnings.filterwarnings("ignore")
# Set global GPU visibility
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def str2bool(v):
    """Convert string argument to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='RUL Prediction Framework')

    # Main Task and Model Configurations
    parser.add_argument('--task', default='PINN', type=str, help='Task options: [normal, PINN]')
    parser.add_argument('--model_name', default='PINN', type=str,
                        help='Model options: [LeNet, LSTM, Transformer, Autoformer, PatchTST, AGCNN, '
                             'Dual_Mixer, FCSTGNN, DegraNet, SDAGCN, PINN, etc.]')
    parser.add_argument('--train', default=False, type=str2bool, help='Train mode if True, else Test mode')
    parser.add_argument('--resume', default=False, type=str2bool, help='Resume from checkpoint')
    parser.add_argument('--save_test', default=True, type=str2bool, help='Save predictions during testing')
    parser.add_argument('--save_path', default=None, type=str, help='Path for result storage')
    parser.add_argument('--resume_path', default=None, type=str, help='Checkpoint path to resume')
    parser.add_argument('--train_epochs', default=150, type=int, help='Total training epochs')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate')
    parser.add_argument('--loss_type', default='MSE', type=str, help='Loss function: [MSE, MAE]')
    parser.add_argument('--info', default='main test', type=str, help='Experimental metadata')

    # Data Loading Parameters
    parser.add_argument('--data_root', default='E:\RUL_Framework11.13', type=str, help='Root directory of datasets')
    parser.add_argument('--dataset_name', default='CMAPSS', type=str, help='Dataset: [CMAPSS, N_CMAPSS, XJTU]')
    parser.add_argument('--Data_id_CMAPSS', default="FD001", type=str, help='Sub-dataset for CMAPSS')
    parser.add_argument('--Data_id_N_CMAPSS', default="DS01", type=str, help='Sub-dataset for N_CMAPSS')
    parser.add_argument('--Data_id_XJTU', default="35", type=str, help='Sub-dataset for XJTU')
    parser.add_argument('--input_length', default=50, type=int, help='Window size (sequence length)')
    parser.add_argument('--validation', default=0.2, type=float, help='Validation split ratio')
    parser.add_argument('--batch_size', default=32, type=int, help='Training batch size')
    parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility')

    # Dataset Specific Parameters
    parser.add_argument('--MAXLIFE_CMAPSS', default=120, type=int, help='Max RUL boundary for CMAPSS')
    parser.add_argument('--normalization_CMAPSS', default="minmax", type=str, help='Normalization: [minmax, zscore]')
    parser.add_argument('--s', type=int, default=10, help='Stride of sliding window')
    parser.add_argument('--sampling', type=int, default=50, help='Downsampling rate')
    parser.add_argument('--change_len', type=str2bool, default=True, help='Regenerate data if input_len changes')
    parser.add_argument('--rate', type=float, default=0.8, help='Max life threshold rate')
    parser.add_argument('--xjtu_n_fea', type=int, default=40, help='Feature dimensions for XJTU')

    # Model Architecture Hyperparameters
    parser.add_argument('--d_model', default=64, type=int, help='Hidden dimension')
    parser.add_argument('--d_ff', default=128, type=int, help='Feed-forward network dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout probability')
    parser.add_argument('--patch_size', type=int, default=15, help='Patch size for PatchTST/Transformer')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')

    # Domain Adaptation & Classification
    parser.add_argument('--DA', default=False, type=str2bool, help='Enable Domain Adaptation')
    parser.add_argument('--source_domain', default="FD001", type=str, help='Source domain ID')
    parser.add_argument('--target_domain', default="FD001", type=str, help='Target domain ID')
    parser.add_argument('--Classify', default=False, type=str2bool, help='Enable auxiliary classification task')

    args = parser.parse_args()

    # Log configurations
    print(f"\n|{'=' * 30} Configuration {'=' * 30}|")
    for key, value in vars(args).items():
        print(f"| {key:>25} : {str(value):<35} |")
    print(f"|{'=' * 75}|\n")

    # Path Setup
    if args.data_root:
        args.data_path = os.path.join(args.data_root, args.dataset_name)
    else:
        args.data_path = os.path.join('.', args.dataset_name)

    # Global Training Settings
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.optimizer = "Adam"
    args.learning_rate_patience = 10
    args.learning_rate_factor = 0.3
    args.early_stop_patience = 3

    # Random Seed Initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Experiment Execution
    if args.task == 'normal':
        exp = Exp_merge(args)
        exp.start()
    elif args.task == 'PINN':
        exp = Exp_PINN(args)
        exp.start()

if __name__ == '__main__':
    main()