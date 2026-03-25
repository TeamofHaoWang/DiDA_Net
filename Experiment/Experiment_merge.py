# -*- coding: utf-8 -*-
import os
import sys
import shutil
import datetime
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Internal imports
from Model import *
from Model.Dual_Mixer import DualMLPMixer
from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from tool.Write_csv import write_csv, write_csv_dict

# Dataset specific loaders
from N_CMAPSS_Related.N_CMAPSS_load_data import get_n_cmapss_data_, N_CMAPSSData_index
from CMAPSS_Related.load_data_CMAPSS import get_cmapss_data_index
from CMAPSS_Related.CMAPSS_Dataset import CMAPSSData_index
from XJTU_Related.load_data_XJTU import get_xjtu_data_
from XJTU_Related.XJTU_Dataset import XJTUData, XJTUData_index


def get_xjtu_config(data_id, base_path='./XJTU/XJTU-SY_Bearing_Datasets'):
    """Helper to get XJTU dataset configurations based on operating conditions."""
    folder_map = {
        "35": "35Hz12kN",
        "37": "37.5Hz11kN",
        "40": "40Hz10kN"
    }
    assert data_id in folder_map, f"Unsupported data_id: {data_id}"
    folder = folder_map[data_id]
    root_dir = os.path.join(base_path, folder)

    if folder == '35Hz12kN':
        train_set, test_set = ['Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5'], ['Bearing1_1']
    elif folder == '37.5Hz11kN':
        train_set, test_set = ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4'], ['Bearing2_5']
    else:  # 40Hz10kN
        train_set, test_set = ['Bearing3_1', 'Bearing3_2', 'Bearing3_4', 'Bearing3_5'], ['Bearing3_3']

    return {"root_dir": root_dir, "train_bearing_data_set": train_set, "test_bearing_data_set": test_set}


class Exp_merge(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_path()

        # Load datasets and loaders
        self.train_data, self.train_loader, self.vali_data, \
        self.vali_loader, self.test_data, self.test_loader, self.input_feature = self._get_data()

        # Initialize model and optimizer mapping
        self.model = self._get_model()
        self.optimizer_dict = {"Adam": optim.Adam}

    def _acquire_device(self):
        """Setup execution device (CPU/GPU)."""
        if self.args.use_gpu:
            device = torch.device('cuda')
            print(f'Using GPU: cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _get_model(self):
        """Model factory based on model_name argument."""
        name = self.args.model_name
        m = None

        if name == 'LeNet':
            m = LeNet(self.args.input_length, self.input_feature)
        elif name == 'DiDA_Net':
            m = DiDA_Net(self.args, self.save_path)
        elif name == 'LSTM':
            m = LSTM(self.input_feature)
        elif name == 'CDSG':
            m = CDSG(self.args)
        elif name == 'SDAGCN':
            m = SDAGCN(self.args, input_feature=self.input_feature)
        elif name == 'Transformer':
            m = Transformer(self.args, input_feature=self.input_feature)
        elif name == 'Autoformer':
            m = Autoformer(self.args, input_feature=self.input_feature)
        elif name == 'PatchTST':
            m = PatchTST(self.args, input_feature=self.input_feature)
        elif name == 'AGCNN':
            m = AGCNN(input_len=self.args.input_length, num_features=self.input_feature, m=15,
                      rnn_hidden_size=[18, 20], dropout_rate=0.2, bidirectional=True, fcn_hidden_size=[20, 10])
        elif name == 'Dual_Mixer':
            m = DualMLPMixer(self.args, self.input_feature)
        elif name == 'FCSTGNN':
            # Simplified FCSTGNN parameter injection
            self._setup_fcstgnn_params()
            m = FC_STGNN_RUL(self.args.patch_size, self.args.conv_out, self.args.lstmhidden_dim,
                             self.args.lstmout_dim, self.args.conv_kernel, self.args.hidden_dim,
                             self.args.conv_time_CNN, self.args.num_sensor, self.args.num_windows,
                             self.args.moving_window, self.args.stride, self.args.decay, self.args.pool_choice, 1)

        if m is None:
            raise ValueError(f"Model {name} not implemented in _get_model")

        print(f"Total Model Parameters: {sum(p.numel() for p in m.parameters())}")
        return m.double().to(self.device)

    def _setup_fcstgnn_params(self):
        """Configuration helper for FCSTGNN model."""
        self.args.k = 1
        self.args.conv_kernel = 2
        self.args.moving_window, self.args.stride = [2, 2], [1, 2]
        self.args.pool_choice, self.args.decay = 'mean', 0.7
        self.args.patch_size, self.args.conv_out = 5, 7
        self.args.hidden_dim = 8
        self.args.lstmhidden_dim = 8
        self.args.num_sensor = self.input_feature
        self.args.num_windows = (self.args.input_length // self.args.patch_size - 1) + \
                                (self.args.input_length // self.args.patch_size // 2)

    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} not supported.")
        return self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

    def _get_data(self):
        """Load data and wrap in DataLoaders based on dataset_name."""
        args = self.args
        if args.dataset_name == 'CMAPSS':
            X_train, y_train, index_train, X_vali, y_vali, index_vali, X_test, y_test, index_test, _ = \
                get_cmapss_data_index(data_path=args.data_path, Data_id=args.Data_id_CMAPSS,
                                      sequence_length=args.input_length, MAXLIFE=args.MAXLIFE_CMAPSS,
                                      validation=args.validation)
            self.max_life = args.MAXLIFE_CMAPSS
        elif args.dataset_name == 'N_CMAPSS':
            X_train, index_train, y_train, X_vali, index_vali, y_vali, X_test, index_test, y_test, self.max_life = \
                get_n_cmapss_data_(args=args, data_path=args.data_path, name=args.Data_id_N_CMAPSS)
        elif args.dataset_name == 'XJTU':
            paths = get_xjtu_config(args.Data_id_XJTU, os.path.join(args.data_path, 'XJTU-SY_Bearing_Datasets'))
            X_train, y_train, X_vali, y_vali, index_train, index_vali, X_test, y_test, index_test, self.max_life = \
                get_xjtu_data_(root_dir=paths['root_dir'], train_bearing_data_set=paths['train_bearing_data_set'],
                               test_bearing_data_set=paths['test_bearing_data_set'], window_length=args.input_length,
                               input_fea=args.xjtu_n_fea, sampling=args.sampling, stride=args.s,
                               max_life_rate=args.rate)
        else:
            raise ValueError(f"Dataset {args.dataset_name} not recognized.")

        # Create dataset instances dynamically
        dataset_cls = eval(f"{args.dataset_name}Data_index")
        train_set = dataset_cls(X_train, index_train, y_train)
        vali_set = dataset_cls(X_vali, index_vali, y_vali)
        test_set = dataset_cls(X_test, index_test, y_test)

        input_fea = X_test.shape[-1]
        args.input_feature = input_fea

        # DataLoader configuration
        loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0}
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
        vali_loader = DataLoader(vali_set, shuffle=False, drop_last=True, **loader_kwargs)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)

        return train_set, train_loader, vali_set, vali_loader, test_set, test_loader, input_fea

    def _get_path(self):
        """Initialize log and result directories."""
        base_log_dir = './logs/'
        if not os.path.exists(base_log_dir): os.makedirs(base_log_dir)

        dataset_id = getattr(self.args, f"Data_id_{self.args.dataset_name}", self.args.dataset_name)
        self.model_path = os.path.join(base_log_dir, dataset_id, self.args.model_name)

        if not os.path.exists(self.model_path): os.makedirs(self.model_path)

        exp_id = self.args.save_path
        if exp_id and exp_id.lower() != 'none':
            self.save_path = os.path.join(self.model_path, exp_id)
            if self.args.train and os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
        else:
            existing = [int(d[3:]) for d in os.listdir(self.model_path) if d.startswith('exp')]
            new_id = max(existing) + 1 if existing else 0
            self.save_path = os.path.join(self.model_path, f'exp{new_id}')

        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.args.save_path = self.save_path

    def save_hparam(self, per_epoch_time):
        """Export hyperparameters and timing to YAML."""
        params = {k: v for k, v in vars(self.args).items() if not k.startswith('_')}
        exclude = ['train', 'resume', 'save_path', 'resume_path', 'batch_size', 'train_epochs', 'learning_rate']
        for k in exclude: params.pop(k, None)

        with open(os.path.join(self.save_path, 'hparam.yaml'), 'w') as f:
            yaml.dump(params, f)
            yaml.dump({'epoch_times': per_epoch_time}, f)

    def start(self):
        """Execution entry point for training and testing."""
        if self.args.train:
            early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
            lr_adapter = adjust_learning_rate_class(self.args, True)
            self.model_optim = self._select_optimizer()

        # Loss selection
        if self.args.loss_type == 'MSE':
            self.loss_criterion = nn.MSELoss()
        elif self.args.loss_type == 'MAE':
            self.loss_criterion = nn.L1Loss()
        elif self.args.loss_type == 'QUAN':
            self.loss_criterion = QuantileLoss(quantile=0.3)
        else:
            raise ValueError("Unsupported loss type.")

        if self.args.resume:
            print('Loading checkpoint...')
            ckpt_path = os.path.join(self.model_path, self.args.resume_path, 'best_checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt_path))

        per_epoch_time = {}
        if self.args.train:
            print("Starting training...")
            for epoch in range(self.args.train_epochs):
                train_loss, epoch_time = self.training()
                per_epoch_time[f"epoch_{epoch}"] = epoch_time
                vali_loss = self.validation(self.vali_loader, self.loss_criterion)

                print(
                    f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} | Time: {epoch_time:.2f}s")

                early_stopping(vali_loss, self.model, self.save_path)
                if early_stopping.early_stop: break
                lr_adapter(self.model_optim, vali_loss)

            # Load best model for testing
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_checkpoint.pth')))

        train_avg_time = np.mean(list(per_epoch_time.values())) if per_epoch_time else 0
        self.save_hparam(per_epoch_time)

        # Final Evaluation
        rmse, overall_rmse, score, test_time = self.test(self.test_loader)
        print(f"Test Performance -> RMSE: {rmse:.4f}, Score: {score:.4f}")

        self._log_to_csv(rmse, overall_rmse, score, train_avg_time, test_time)

    def training(self):
        start_time = time()
        train_loss = []
        self.model.train()
        for i, (batch_x, idx_x, batch_y) in enumerate(tqdm(self.train_loader, desc="Training")):
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.double().to(self.device), batch_y.double().to(self.device)

            # Normalize target for CMAPSS/N-CMAPSS
            target = batch_y if self.args.dataset_name == 'XJTU' else batch_y / self.max_life
            _, outputs = self.model(batch_x, mode='train', idx=idx_x)

            loss = self.loss_criterion(outputs, target)
            loss.backward()
            self.model_optim.step()
            train_loss.append(loss.item())

        return np.mean(train_loss), time() - start_time

    def validation(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, idx_x, batch_y in vali_loader:
                batch_x, batch_y = batch_x.double().to(self.device), batch_y.double().to(self.device)
                target = batch_y if self.args.dataset_name == 'XJTU' else batch_y / self.max_life
                _, outputs = self.model(batch_x, mode='val', idx=idx_x)
                total_loss.append(self.loss_criterion(outputs, target).item())
        return np.mean(total_loss)

    def test(self, test_loader):
        self.model.eval()
        preds, trues = [], []
        total_inf_time = 0.0

        with torch.no_grad():
            for batch_x, idx_x, batch_y in tqdm(test_loader, desc="Testing"):
                batch_x = batch_x.to(self.device).double()

                start = time()
                _, outputs = self.model(batch_x, mode='test', idx=idx_x)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                total_inf_time += (time() - start)

                # Post-processing
                if self.args.is_minmax: outputs = outputs * self.max_life

                out_np = outputs.cpu().numpy()
                gt_np = batch_y.numpy()

                if self.args.dataset_name == 'XJTU' and self.args.is_minmax:
                    out_np, gt_np = out_np / self.max_life, gt_np / self.max_life

                preds.append(out_np)
                trues.append(gt_np)

        avg_test_batch_time = total_inf_time / len(test_loader)
        preds = np.concatenate(preds).reshape(-1, 1)
        trues = np.concatenate(trues).reshape(-1, 1)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(trues)), preds, label='Pred', alpha=0.6)
        plt.scatter(range(len(trues)), trues, label='GT', alpha=0.6)
        plt.legend();
        plt.title(f"Test Results - {self.args.model_name}")
        plt.savefig(os.path.join(self.save_path, 'visual_result.png'))
        plt.close()

        if self.args.save_test:
            np.savez(os.path.join(self.save_path, 'result.npz'), preds=preds, trues=trues)

        rmse = np.sqrt(mean_squared_error(preds, trues))
        score = self.score_compute(preds, trues)
        return rmse, rmse, score, avg_test_batch_time

    def score_compute(self, pred, gt):
        """Custom RUL scoring function."""
        if self.args.dataset_name == 'XJTU':
            idx = np.where(gt != 0.0)[0]
            p, g = pred[idx], gt[idx]
            score_list = np.where(p - g < 0, np.exp((g - p) * 100 * np.log(0.5) / (g * 20)),
                                  np.exp(-(g - p) * 100 * np.log(0.5) / (g * 5)))
            return np.mean(score_list)
        else:
            diff = pred - gt
            score_list = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
            return np.sum(score_list) if self.args.dataset_name == 'CMAPSS' else np.mean(score_list)

    def _log_to_csv(self, rmse, overall_rmse, score, t_avg, inf_avg):
        """Log final metrics to a centralized CSV file."""
        log_path = './logs/efficiency_experimental_logs.csv'
        if not os.path.exists(log_path):
            header = [
                ['dataset', 'model', 'time', 'LR', 'batch_size', 'RMSE', 'Score', 'params', 'train_time', 'test_time',
                 'path']]
            write_csv(log_path, header, 'w+')

        dataset_name = getattr(self.args, f"Data_id_{self.args.dataset_name}", self.args.dataset_name)
        log_entry = [{
            'dataset': dataset_name, 'model': self.args.model_name,
            'time': datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            'LR': self.args.learning_rate, 'batch_size': self.args.batch_size,
            'RMSE': rmse, 'Score': score,
            'params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'train_time': t_avg, 'test_time': inf_avg, 'path': self.save_path
        }]
        write_csv_dict(log_path, log_entry, 'a+')


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantile=0.1):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        errors = target - pred
        return torch.mean(torch.max((self.quantile - 1) * errors, self.quantile * errors))