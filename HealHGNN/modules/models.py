import torch.nn as nn
import torch.nn.functional as F
from utils.train_utils import ActivateModule, NormModule
from modules.layers import HGNNlayer1,HGNNlayer2,HEALlayer
from torch_geometric.utils import add_self_loops
import torch
import torch_scatter
import numpy as np
from torch_scatter import scatter_sum,scatter
class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def flops(self, x):
        num_samples = np.prod(x.shape[:-1])
        flops = num_samples * self.in_channels # first normalization
        flops += num_samples * self.in_channels * self.hidden_channels # first linear layer
        flops += num_samples * self.hidden_channels # first relu layer

        # flops for each layer
        per_layer = num_samples * self.hidden_channels * self.hidden_channels
        per_layer += num_samples * self.hidden_channels # relu + normalization
        flops += per_layer * (len(self.lins) - 2)

        flops += num_samples * self.out_channels * self.hidden_channels # last linear layer

        return flops



class HEAL(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, embed_dim, out_dim,
                 bias, input_act, act, drop=0.3, norm='ln',
                 add_self_loop=True, tau=0.1,m=1,n=1, layer_wise=True):
        super().__init__()
        self.input_lin = nn.Sequential(nn.Linear(in_dim, embed_dim, bias=bias),
                                       nn.Dropout(drop),
                                       ActivateModule(input_act))
        self.rate1 = nn.Sequential(nn.Linear(embed_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  ActivateModule(act),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  NormModule(norm, hid_dim)
                                  ) if layer_wise else None
        self.rate2 = nn.Sequential(nn.Linear(embed_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  ActivateModule(act),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  NormModule(norm, hid_dim)
                                  ) if layer_wise else None
        self.w1 = nn.Sequential(nn.Linear(embed_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if layer_wise else None
        self.w2 = nn.Sequential(nn.Linear(embed_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if layer_wise else None
        self.fc1 = None
        self.fc2 = None
        self.gamma1 = None
        self.gamma2 = None
        self.layers = nn.ModuleList([HEALlayer(embed_dim, hid_dim, embed_dim,bias, act, drop,norm,
                                                            self.rate1,self.rate2,self.gamma1,self.gamma2,self.w1,self.w2, self.fc1, self.fc2,m=m,n=n) for _ in range(n_layers)])
        self.out_norm = NormModule(norm, embed_dim)
        self.norm = NormModule(norm, embed_dim)
        self.out_lin = nn.Linear(embed_dim, out_dim, bias=bias)
        self.drop = nn.Dropout(drop)
        self.add_self_loop = add_self_loop
        self.indv_layer = HGNNlayer1(embed_dim, hid_dim, 1, bias=bias, drop=drop)
        self.inde_layer = HGNNlayer2(embed_dim,hid_dim, 1, bias=bias, drop=drop)
        self.tau = tau
    def forward(self, data):
        self.deg_cal(data)
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        x0v = self.input_lin(x)
        x0e = self.trans(x0v,V,E)
        x0v = self.norm(x0v)
        x0e = self.norm(x0e)
        xtv = x0v
        xte = x0e
        # edge_index = add_self_loops(data.edge_index)[0] if self.add_self_loop else data.edge_index
        ind_bdv = F.logsigmoid(self.indv_layer(x0v, V,E) / self.tau).exp()
        ind_bde = F.logsigmoid(self.inde_layer(x0e, V,E) / self.tau).exp()
        for layer in self.layers:
            xtv,xte = layer(xtv,xte, x0v, x0e,V,E, ind_bdv,ind_bde,self.deg_v_inv_sqrt,self.deg_e_inv_sqrt,self.deg_v_inv,self.deg_e_inv)
        x = self.out_lin(xtv)
        return x
    
    def trans(self,x,V,E):
        num_nodes = x.size(0)
        num_edges = E.max().item() + 1
        x = x * self.deg_v_inv_sqrt
        x_edge = scatter_sum(x[V], E, dim=0, dim_size=num_edges)
        out = x_edge * self.deg_e_inv_sqrt
        return out
    
    def deg_cal(self,data):
        V, E = data.edge_index[0], data.edge_index[1]
        num_nodes = data.num_nodes
        num_edges = E.max().item() + 1
        
        deg_v = scatter_sum(torch.ones_like(V, dtype=torch.float), V, dim=0, dim_size=num_nodes)
        deg_v_inv = deg_v.pow(-1)
        deg_v_inv[deg_v_inv == float('inf')] = 0
        deg_v_inv_sqrt = deg_v.pow(-0.5)
        deg_v_inv_sqrt[deg_v_inv_sqrt == float('inf')] = 0

        deg_e = scatter_sum(torch.ones_like(E, dtype=torch.float), E, dim=0, dim_size=num_edges)
        deg_e_inv = deg_e.pow(-1)
        deg_e_inv[deg_e_inv == float('inf')] = 0
        deg_e_inv_sqrt = deg_e.pow(-0.5)
        deg_e_inv_sqrt[deg_e_inv_sqrt == float('inf')] = 0

        # restore to buffer
        self.register_buffer("deg_v_inv_sqrt", deg_v_inv_sqrt.unsqueeze(1))
        self.register_buffer("deg_e_inv_sqrt", deg_e_inv_sqrt.unsqueeze(1))
        self.register_buffer("deg_v_inv", deg_v_inv.unsqueeze(1))
        self.register_buffer("deg_e_inv", deg_e_inv.unsqueeze(1))
        self.register_buffer("V", V)
        self.register_buffer("E", E)