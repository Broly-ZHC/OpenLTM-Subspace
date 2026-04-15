import os
import time
import datetime
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        
    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            self.device = self.args.gpu
        
        model = self.model_dict[self.args.model].Model(self.args)
        
        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
            
        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        
        self.model.eval()    
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if is_test or self.args.nonautoregressive:
                        outputs = outputs[:, -self.args.output_token_len:, :]
                        batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
            
        if self.args.model == 'gpt4ts':
            # GPT4TS just requires to train partial layers
            self.model.in_layer.train()
            self.model.out_layer.train()
        else: 
            self.model.train()
            
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        
        # Collect per-epoch statistics for the result file
        self._epoch_stats = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            epoch_train_loss = 0.0
            epoch_train_count = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if self.args.dp:
                    torch.cuda.synchronize()
                if self.args.nonautoregressive:
                    batch_y = batch_y[:, -self.args.output_token_len:, :]
                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                # ---- Auxiliary Loss (Subspace Clustering / VQ) ----
                main_loss = loss
                inner_model = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model
                if hasattr(inner_model, '_aux_loss') and inner_model._aux_loss is not None:
                    aux_weight = getattr(self.args, 'vq_beta', 1.0)
                    total_loss = main_loss + aux_weight * inner_model._aux_loss
                else:
                    total_loss = main_loss
                # --------------------------------------------------

                # Accumulate batch loss for epoch-level average
                epoch_train_loss += loss.item() * batch_x.shape[0]
                epoch_train_count += batch_x.shape[0]

                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                total_loss.backward()
                model_optim.step()

            epoch_train_time = time.time() - epoch_time
            avg_train_loss = epoch_train_loss / max(epoch_train_count, 1)

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, epoch_train_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)

            self._epoch_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'vali_loss': float(vali_loss),
                'epoch_time': epoch_train_time,
            })
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            checkpoint = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            for name, param in self.model.named_parameters():
                if not param.requires_grad and name not in checkpoint:
                    checkpoint[name] = param
            self.model.load_state_dict(checkpoint)
            
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                inference_steps = self.args.test_pred_len // self.args.output_token_len
                dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):  
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)
                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.output_token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-self.args.output_token_len+dis, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                if self.args.visualize and i % 2 == 0:
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)
        if self.args.covariate:
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # ---- Determine output path and filename ----
        def _normalize_model_name(name: str) -> str:
            return ''.join(ch for ch in name.lower() if ch.isalnum())

        def _infer_dataset() -> str:
            rp = getattr(self.args, 'root_path', '') or ''
            rp_norm = rp.replace('\\', '/').rstrip('/')
            base = os.path.basename(rp_norm) if rp_norm else ''
            return base or (
                getattr(self.args, 'data_path', '').split('.')[0]
                or getattr(self.args, 'data', 'dataset')
            )

        model_name      = _normalize_model_name(getattr(self.args, 'model', 'model'))
        dataset_name    = _infer_dataset()
        task_name       = getattr(self.args, 'task_name', 'task')
        seq_len         = int(getattr(self.args, 'seq_len', 0))
        input_token_len = int(getattr(self.args, 'input_token_len', seq_len))
        pred_len        = int(getattr(self.args, 'test_pred_len', 0))
        num_groups      = int(getattr(self.args, 'num_groups', 16))
        enc_in          = int(getattr(self.args, 'enc_in', 0))

        # "features": M = multivariate (enc_in > 1), S = univariate
        features = 'M' if enc_in > 1 else 'S'

        # model_id: use the one from run.py args, else reconstruct
        model_id = getattr(
            self.args, 'model_id',
            f'{dataset_name}_{input_token_len}_{pred_len}'
        )

        # Path: ./result/{dataset}/{model}/model_id_numgroups.txt
        result_dir = os.path.join('./result', dataset_name, model_name)
        os.makedirs(result_dir, exist_ok=True)
        file_path = os.path.join(result_dir, f"{model_id}_{num_groups}.txt")

        # ---- Collect epoch statistics ----
        epoch_stats     = getattr(self, '_epoch_stats', [])
        train_time_cost = getattr(self, '_train_time_cost', None)

        # Average batch time (total_train_time / epochs_ran)
        if train_time_cost and epoch_stats:
            total_epochs_ran = len(epoch_stats)
            avg_batch_time = train_time_cost / total_epochs_ran
        else:
            avg_batch_time = None

        # Count existing runs to assign run number
        run_no = 1
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as _rf:
                    run_no = sum(
                        1 for line in _rf
                        if line.startswith('textmodel_id:')
                    ) + 1
            except Exception:
                run_no = 1

        total_time_seconds = float(train_time_cost) if train_time_cost else 0.0
        total_train_epochs_time = sum(s['epoch_time'] for s in epoch_stats) if epoch_stats else 0.0

        with open(file_path, 'a') as f:
            # ---- Core metrics block ----
            f.write(f"textmodel_id: {model_id}\n")
            f.write(f"dataset: {dataset_name}\n")
            f.write(f"model: {model_name}\n")
            f.write(f"seq_len: {seq_len}\n")
            f.write(f"pred_len: {pred_len}\n")
            f.write(f"features: {features}\n")
            f.write(f"mse: {mse:.11g}\n")
            f.write(f"mae: {mae:.11g}\n")
            f.write(f"rmse: {rmse:.11g}\n")
            f.write(f"mape: {mape:.11g}\n")
            f.write(f"mspe: {mspe:.11g}\n")
            f.write(f"total_time_seconds: {total_time_seconds:.2f}\n")
            if avg_batch_time is not None:
                f.write(f"avg_batch_time_seconds: {avg_batch_time:.6f}\n")
            f.write(f"\n")
            # ---- Epoch details block ----
            f.write(f"# Setting: {task_name}\n")
            f.write(f"# Experiment ID: {model_id}\n")
            f.write(f"# Description: Exp\n")
            f.write(f"# Epoch Details:\n")
            f.write(f"# Epochs completed: {len(epoch_stats)}\n")
            f.write(f"# Total train epochs time: {total_train_epochs_time:.2f}s\n")
            f.write(f"# Per-Epoch Time (seconds):\n")
            for s in epoch_stats:
                f.write(f"epoch_{s['epoch']}_time: {s['epoch_time']:.4f}\n")
            f.write(f"\n")
            f.write(f"# Per-Epoch Train Loss:\n")
            for s in epoch_stats:
                f.write(f"epoch_{s['epoch']}_train_loss: {s['train_loss']:.6f}\n")
            f.write(f"\n")
            f.write(f"# Per-Epoch Validation Loss:\n")
            for s in epoch_stats:
                f.write(f"epoch_{s['epoch']}_vali_loss: {s['vali_loss']:.6f}\n")
            f.write(f"\n")
        return
