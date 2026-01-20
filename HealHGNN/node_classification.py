import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from utils.eval_utils import cal_accuracy, cal_F1
from utils.train_utils import EarlyStopping
from logger import create_logger
import os
from modules.models import HEAL
from torch_geometric.utils import degree, is_undirected, to_undirected
import utils.dataset as dataset
import torch_geometric
import utils.utils as utils
class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger = create_logger(configs.log_path)

    def load_model(self, dataset):
        nc_model = None
        if self.configs.method == 'HEAL':
            nc_model = HEAL(n_layers=self.configs.n_layers,
                       in_dim=dataset.num_features, hid_dim=self.configs.hid_dim,
                               embed_dim=self.configs.embed_dim, out_dim=dataset.num_classes,
                               bias=self.configs.bias, act=self.configs.act, input_act=self.configs.input_act,
                               drop=self.configs.dropout, norm=self.configs.norm,
                               add_self_loop=self.configs.add_self_loop,
                               tau=self.configs.tau,m=self.configs.u,n=self.configs.lam, layer_wise=self.configs.layer_wise).to(self.device)

        return nc_model

    def load_data(self):
        if self.configs.add_self_loop:
            transform = torch_geometric.transforms.Compose([dataset.AddHypergraphSelfLoops()])
        else:
            transform = None
        root = os.path.join(self.configs.data_dir, self.configs.dataset)
        path_to_download = os.path.join(self.configs.raw_data_dir, self.configs.dataset)
        data = dataset.HypergraphDataset(root=root, name=self.configs.dataset, path_to_download=path_to_download,
        feature_noise=self.configs.feature_noise, transform=transform).data
        return data
    
    def get_split_idx(self,configs,data):
        split_idx_lst = []
        for run in range(configs.exp_iters):
            split_idx = utils.rand_train_test_idx(
                data.y, train_prop=configs.train_prop, valid_prop=configs.valid_prop)
            split_idx_lst.append(split_idx)
        return split_idx_lst



    def train(self):
        total_test_acc = []
        total_test_weighted_f1 = []
        total_test_macro_f1 = []
        with open(self.configs.result_path, 'a') as f:
            f.write(f"---------------------{self.configs.dataset}--------------------------\n")
            f.write(f"{self.configs}\n")
        self.logger.info("--------------------------Training Start-------------------------")
        data = self.load_data()
        print(data.x.shape)
        print(data.edge_index[1].max()+1)
        
        # count_nested_hyperedges(data)
        split_idx_lst = self.get_split_idx(self.configs,data)
        for t in range(self.configs.exp_iters):
            split_idx = split_idx_lst[t]
            train_idx = split_idx['train'].to(self.device)
            nc_model = self.load_model(data)
            nc_model.train()
            optimizer = Adam(nc_model.parameters(), lr=self.configs.lr_nc,
                             weight_decay=self.configs.weight_decay_nc)
            early_stop = EarlyStopping(self.configs.patience_nc)
            for epoch in range(self.configs.epochs_nc):
                data = data.to(self.device)
                train_loss, pred, true = self.train_step(nc_model, data, optimizer,train_idx)
                train_acc = cal_accuracy(pred, true)
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

                if epoch % self.configs.val_every == 0:
                    val_loss, val_acc, val_weighted_f1, val_macro_f1 = self.val(nc_model, data, split_idx['valid'])
                    self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, "
                                     f"val_acc={val_acc * 100: .2f}%,"
                                     f"val_weighted_f1={val_weighted_f1 * 100: .2f},"
                                     f"val_macro_f1={val_macro_f1 * 100: .2f}%")
                    early_stop(-val_acc, nc_model, self.configs.checkpoints, self.configs.task_model_path)
                    if early_stop.early_stop:
                        print("---------Early stopping--------")
                        break
            test_acc, weighted_f1, macro_f1 = self.test(nc_model, split_idx['test'], data=data)
            self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                             f"weighted_f1={weighted_f1 * 100: .2f},"
                             f"macro_f1={macro_f1 * 100: .2f}%")
            with open(self.configs.result_path, 'a') as f:
                f.write(f"Iter {t}: ACC={test_acc * 100: .2f}%\n")
            total_test_acc.append(test_acc)
            total_test_weighted_f1.append(weighted_f1)
            total_test_macro_f1.append(macro_f1)
        mean, std = np.mean(total_test_acc), np.std(total_test_acc)
        res_str = (f"Best ACCs: {[round(acc * 100, 2) for acc in total_test_acc]}\n" +
                   f"Evaluation Acc is {mean * 100: .2f}% \u00B1 {std * 100: .2f}%\n")
        self.logger.info(res_str)
        with open(self.configs.result_path, 'a') as f:
            f.write(res_str)
        f.close()

    def val(self, nc_model, data,split_idx):
        loss, pred, true = self.test_step(nc_model, data,split_idx)
        acc = cal_accuracy(pred, true)
        weighted_f1, macro_f1 = cal_F1(pred, true)
        nc_model.train()
        return loss, acc, weighted_f1, macro_f1

    def test(self, nc_model,split_idx, data=None):
        data = self.load_data()[1] if data is None else data
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        nc_model.load_state_dict(torch.load(path))
        _, pred, true = self.test_step(nc_model, data, split_idx)
        test_acc = cal_accuracy(pred, true)
        weighted_f1, macro_f1 = cal_F1(pred, true)
        self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                         f"weighted_f1={weighted_f1 * 100: .2f},"
                         f"macro_f1={macro_f1 * 100: .2f}%")
        return test_acc, weighted_f1, macro_f1

    def train_step(self, nc_model, data, optimizer,train_idx):
        optimizer.zero_grad()
        out = nc_model(data)
        loss, pred, true = self.cal_loss(out, data.y, train_idx)
        loss.backward()
        optimizer.step()
        return loss.item(), pred, true

    def test_step(self, nc_model, data, mask):
        nc_model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = nc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, mask)
        return loss.item(), pred, true

    @staticmethod
    def cal_loss(output, label, mask):
        out = output[mask]
        y = label[mask].reshape(-1)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=-1).detach().cpu().numpy()
        return loss, pred, y.detach().cpu().numpy()