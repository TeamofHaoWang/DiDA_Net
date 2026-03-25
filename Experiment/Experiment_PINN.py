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
from Model.DA_LSTM import DA_LSTM
from Model.Dual_Mixer import DualMLPMixer
from layers.relobralo import relobralo
from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from tool.Write_csv import write_csv, write_csv_dict

# Dataset specific loaders (PINN versions)
from N_CMAPSS_Related.N_CMAPSS_load_data import get_n_cmapss_data_PINN, N_CMAPSSData_index_PINN
from CMAPSS_Related.load_data_CMAPSS import get_cmapss_data_index_PINN
from CMAPSS_Related.CMAPSS_Dataset import CMAPSSData_index_PINN
from XJTU_Related.load_data_XJTU import get_xjtu_data_PINN
from XJTU_Related.XJTU_Dataset import XJTUData_index_PINN

# Global Seed for Reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.autograd.set_detect_anomaly(True)


def get_xjtu_config(data_id, base_path='./XJTU/XJTU-SY_Bearing_Datasets'):
    """Config helper for XJTU dataset variations."""
    folder_map = {"35": "35Hz12kN", "37": "37.5Hz11kN", "40": "40Hz10kN"}
    assert data_id in folder_map, f"Unsupported data_id: {data_id}"
    folder = folder_map[data_id]
    root_dir = os.path.join(base_path, folder)

    if folder == '35Hz12kN':
        train_set, test_set = ['Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5'], ['Bearing1_1']
    elif folder == '37.5Hz11kN':
        train_set, test_set = ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4'], ['Bearing2_5']
    else:
        train_set, test_set = ['Bearing3_1', 'Bearing3_2', 'Bearing3_4', 'Bearing3_5'], ['Bearing3_3']
    return {"root_dir": root_dir, "train_bearing_data_set": train_set, "test_bearing_data_set": test_set}


class Exp_PINN(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_path()

        # Load data, build model, and set optimizer
        self.train_data, self.train_loader, self.vali_data, \
        self.vali_loader, self.test_data, self.test_loader, self.input_feature = self._get_data()

        self.model = self._get_model()
        self.optimizer_dict = {"Adam": optim.Adam}

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device

    def _get_model(self):
        """Initialize PINN model and dynamic loss parameters."""
        if self.args.model_name == 'PINN':
            # Parameters for ReLOBRALO (Adaptive Loss Weighting)
            self.r = (np.random.uniform(size=int(1e8)) < 0.9999).astype(float)
            self.a = [1, 0, 0.999]
            self.l0, self.l1 = [1, 1], [1, 1]
            self.lamb = [1, 1]
            self.coef = 100  # Scaling coefficient for physics loss
            model = PINN(self.args, hidden_dim=3, derivatives_order=2)

        print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
        return model.double().to(self.device)

    def _get_data(self):
        """Data loading entry point for PINN tasks."""
        args = self.args
        if args.dataset_name == 'CMAPSS':
            X_train, y_train, _, X_vali, y_vali, _, X_test, y_test, index_test, _ = \
                get_cmapss_data_index_PINN(data_path=args.data_path, Data_id=args.Data_id_CMAPSS,
                                           sequence_length=args.input_length, MAXLIFE=args.MAXLIFE_CMAPSS,
                                           validation=args.validation)
            self.max_life = args.MAXLIFE_CMAPSS
        elif args.dataset_name == 'N_CMAPSS':
            X_train, index_train, y_train, X_vali, index_vali, y_vali, X_test, index_test, y_test, self.max_life = \
                get_n_cmapss_data_PINN(args=args, data_path=args.data_path, name=args.Data_id_N_CMAPSS)
        elif args.dataset_name == 'XJTU':
            paths = get_xjtu_config(args.Data_id_XJTU, os.path.join(args.data_path, 'XJTU-SY_Bearing_Datasets'))
            X_train, y_train, X_vali, y_vali, index_train, index_vali, X_test, y_test, index_test, self.max_life = \
                get_xjtu_data_PINN(root_dir=paths['root_dir'], train_bearing_data_set=paths['train_bearing_data_set'],
                                   test_bearing_data_set=paths['test_bearing_data_set'],
                                   window_length=args.input_length,
                                   input_fea=args.xjtu_n_fea, sampling=args.sampling, stride=args.s,
                                   max_life_rate=args.rate)
        else:
            raise ValueError("Dataset not supported for PINN task.")

        # Re-fetch index variables if not provided by specific loaders
        idx_train = index_train if 'index_train' in locals() else None
        idx_vali = index_vali if 'index_vali' in locals() else None

        dataset_cls = eval(f"{args.dataset_name}Data_index_PINN")
        train_set = dataset_cls(X_train, idx_train, y_train)
        vali_set = dataset_cls(X_vali, idx_vali, y_vali)
        test_set = dataset_cls(X_test, index_test, y_test)

        args.input_feature = X_test.shape[-1]
        loader_args = {'batch_size': args.batch_size, 'num_workers': 0}

        return train_set, DataLoader(train_set, shuffle=True, drop_last=True, **loader_args), \
               vali_set, DataLoader(vali_set, shuffle=False, drop_last=True, **loader_args), \
               test_set, DataLoader(test_set, shuffle=False, drop_last=False, **loader_args), \
               args.input_feature

    def _get_path(self):
        """Setup logging and model saving directories."""
        if not os.path.exists('./logs/'): os.makedirs('./logs/')
        dataset_label = getattr(self.args, f"Data_id_{self.args.dataset_name}", self.args.dataset_name)

        self.model_path = os.path.join('./logs/', dataset_label, self.args.model_name)
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)

        if self.args.save_path and self.args.save_path.lower() != 'none':
            self.save_path = os.path.join(self.model_path, self.args.save_path)
            if self.args.train and os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
        else:
            exps = [int(x[3:]) for x in os.listdir(self.model_path) if x.startswith('exp')]
            self.save_path = os.path.join(self.model_path, f'exp{max(exps) + 1 if exps else 0}')

        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.args.save_path = self.save_path

    def _select_optimizer(self):
        return self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)

    def start(self):
        """Main execution flow for PINN experiment."""
        if self.args.train:
            self.model_optim = self._select_optimizer()
            early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
            lr_adapter = adjust_learning_rate_class(self.args, True)

        # Loss function setup
        if self.args.loss_type == 'MSE':
            self.loss_criterion = nn.MSELoss()
        elif self.args.loss_type == 'MAE':
            self.loss_criterion = nn.L1Loss()
        else:
            self.loss_criterion = QuantileLoss(quantile=0.3)

        if self.args.resume:
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_path, self.args.resume_path, 'best_checkpoint.pth')))

        per_epoch_time = {}
        if self.args.train:
            print("Starting PINN Training...")
            for epoch in range(self.args.train_epochs):
                train_loss, epoch_time = self.training(epoch)
                per_epoch_time[f"epoch_{epoch}"] = epoch_time
                vali_loss = self.validation(self.vali_loader)

                print(
                    f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | Time: {epoch_time:.2f}s")

                early_stopping(vali_loss, self.model, self.save_path)
                if early_stopping.early_stop: break
                lr_adapter(self.model_optim, vali_loss)

            self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_checkpoint.pth')))

        # Final testing and logging
        avg_train_time = np.mean(list(per_epoch_time.values())) if per_epoch_time else 0
        rmse, _, score, test_time = self.test(self.test_loader)
        self._log_results(rmse, score, avg_train_time, test_time)

    def training(self, epoch):
        self.model.train()
        start_time = time()
        losses_total = []

        for i, (batch_x, idx_x, batch_y) in enumerate(tqdm(self.train_loader, desc="Training")):
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.double().to(self.device), batch_y.double().to(self.device)

            # PINN output: predictions, hidden states, and physics residuals (f)
            outputs, h, f = self.model(batch_x, mode='train', idx=idx_x)

            # Normalization logic
            target = batch_y if self.args.dataset_name == 'XJTU' else batch_y / self.max_life

            # Data Loss (u) and Physics Loss (f)
            loss_u = torch.sqrt(self.loss_criterion(outputs, target))
            loss_f = torch.sqrt(self.loss_criterion(f, torch.zeros_like(f)))

            # Adaptive weighting using ReLOBRALO
            self.lamb = relobralo(loss_u=loss_u, loss_f=self.coef * loss_f, alpha=self.a[0],
                                  l0=self.l0, l1=self.l1, lam=self.lamb, T=0.1, rho=self.r[0])

            loss = self.lamb[0] * loss_u + self.lamb[1] * self.coef * loss_f

            # Update ReLOBRALO memory
            if len(self.a) > 1: self.a = self.a[1:]
            self.r = self.r[1:]
            current_losses = [loss_u.item(), (self.coef * loss_f).item()]
            if epoch == 0 and i == 0: self.l0 = current_losses
            self.l1 = current_losses

            loss.backward()
            self.model_optim.step()
            losses_total.append(loss.item())

        return np.mean(losses_total), time() - start_time

    def validation(self, vali_loader):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, idx_x, batch_y in vali_loader:
                batch_x, batch_y = batch_x.double().to(self.device), batch_y.double().to(self.device)
                target = batch_y if self.args.dataset_name == 'XJTU' else batch_y / self.max_life
                outputs, _, _ = self.model(batch_x, mode='val', idx=idx_x)
                total_loss.append(self.loss_criterion(outputs, target).item())
        return np.mean(total_loss)

    def test(self, test_loader):
        # Warm-up phase
        self.model.eval()
        print("Warm-up phase...")
        for i, (bx, idx, _) in enumerate(test_loader):
            if i >= 10: break
            self.model(bx.double().to(self.device), mode='test', idx=idx)
            if self.device.type == 'cuda': torch.cuda.synchronize()

        # Actual inference measurement
        preds, trues = [], []
        inf_time = 0.0
        with torch.no_grad():
            for batch_x, idx_x, batch_y in tqdm(test_loader, desc="Testing"):
                batch_x = batch_x.double().to(self.device)

                t_start = time()
                outputs, _, _ = self.model(batch_x, mode='test', idx=idx_x)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                inf_time += (time() - t_start)

                if self.args.is_minmax: outputs = outputs * self.max_life

                out_np, gt_np = outputs.cpu().numpy(), batch_y.numpy()
                if self.args.dataset_name == 'XJTU' and self.args.is_minmax:
                    out_np, gt_np = out_np / self.max_life, gt_np / self.max_life

                preds.append(out_np)
                trues.append(gt_np)

        avg_batch_time = inf_time / len(test_loader)
        preds, trues = np.concatenate(preds).reshape(-1, 1), np.concatenate(trues).reshape(-1, 1)

        # Plotting & Saving
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(trues)), preds, label='Pred', color='blue', alpha=0.5)
        plt.scatter(range(len(trues)), trues, label='GT', color='red', alpha=0.5)
        plt.legend();
        plt.title(f"PINN Result - {self.args.dataset_name}")
        plt.savefig(os.path.join(self.save_path, 'visual_result.png'))
        plt.close()

        if self.args.save_test:
            np.savez(os.path.join(self.save_path, 'result.npz'), test_preds=preds, test_trues=trues)

        rmse = np.sqrt(mean_squared_error(preds, trues))
        score = self.score_compute(preds, trues)
        return rmse, rmse, score, avg_batch_time

    def score_compute(self, pred, gt):
        """RUL Scoring logic for PINN evaluation."""
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

    def _log_results(self, rmse, score, train_avg, test_avg):
        """Final CSV logging."""
        log_path = './logs/warm_efficiency_experimental_logs.csv'
        if not os.path.exists(log_path):
            header = [['dataset', 'model', 'time', 'RMSE', 'Score', 'params', 'train_avg', 'test_avg', 'path']]
            write_csv(log_path, header, 'w+')

        dataset_id = getattr(self.args, f"Data_id_{self.args.dataset_name}", self.args.dataset_name)
        log_data = [{
            'dataset': dataset_id, 'model': self.args.model_name,
            'time': datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            'RMSE': rmse, 'Score': score,
            'params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'train_average_time': train_avg, 'test_batch_time': test_avg,
            'savepath': self.save_path
        }]
        write_csv_dict(log_path, log_data, 'a+')


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.1):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        errors = target - pred
        return torch.mean(torch.max((self.quantile - 1) * errors, self.quantile * errors))