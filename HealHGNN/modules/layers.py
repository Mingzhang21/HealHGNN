import torch.nn as nn
import torch
from torch_scatter import scatter_sum,scatter,scatter_mean
from torch_geometric.utils import normalize_edge_index
from utils.train_utils import ActivateModule, NormModule
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act='gelu', drop=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            nn.Dropout(drop),
            ActivateModule(act),
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class HEALlayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act='gelu', drop=0.3, norm='ln',
                 rate1=None,rate2=None,gamma1=None,gamma2=None,w1=None,w2=None,fc1=None,fc2=None,m=1.0,n=1.0):
        super().__init__()
        self.m = m
        self.n = n
        self.rate1 = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  ActivateModule(act),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  NormModule(norm, hid_dim)
                                  ) if rate1 is None else rate1
        self.rate2 = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  ActivateModule(act),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  NormModule(norm, hid_dim)
                                  ) if rate2 is None else rate2
        self.gamma1 = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if gamma1 is None else gamma1 
        self.gamma2 = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if gamma2 is None else gamma2 
        self.w1 = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if w1 is None else w1
        self.w2 = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if w2 is None else w2
        self.norm = NormModule(norm, hid_dim)
        self.fc1 = FeedForwardLayer(hid_dim, hid_dim, out_dim, bias, act, drop)if fc1 is None else fc1
        self.fc2 = FeedForwardLayer(hid_dim, hid_dim, out_dim, bias, act, drop)if fc2 is None else fc2
    def forward(self, xtv,xte, x0v,x0e,V,E, ind_bdv,ind_bde, deg_v_inv_sqrt,deg_e_inv_sqrt,deg_v_inv,deg_e_inv):
        """
        x : N * D
        edge_index: 2 * E
        degrees: N
        """
        num_nodes = xtv.shape[0]
        num_edges = E.max().item() + 1
        rate1 = self.rate1(xtv)
        rate2 = self.rate2(xte)
        gamma1 = self.gamma1(x0v)
        gamma2 = self.gamma2(x0e)
        w1 = self.w1(xtv)
        w2 = self.w2(xte)
        ind_bdv = ind_bdv.view(-1) 
        ind_intv = 1 - ind_bdv
        ind_bde = ind_bde.view(-1) 
        ind_inte = 1 - ind_bde
        ind_intv = ind_intv.view(-1, 1)
        ind_bdv = ind_bdv.view(-1, 1)
        ind_inte = ind_inte.view(-1, 1)
        ind_bde = ind_bde.view(-1, 1)
        plus_xv = xte*deg_e_inv_sqrt
        plus_xv = self.m*scatter_sum(plus_xv[E], V, dim=0, dim_size=num_nodes)*deg_v_inv_sqrt
        in_xv = ind_intv*self.aggregate(((1 + ind_intv) * xtv),deg_v_inv_sqrt,deg_e_inv ,V,E,num_nodes,num_edges)
        out_xv = rate1*ind_bdv*self.aggregate(ind_intv * xtv ,deg_v_inv_sqrt,deg_e_inv,V,E,num_nodes,num_edges) 
        xv = in_xv+out_xv+ind_intv*plus_xv+w1*ind_bdv*plus_xv                            
        xv = self.fc1(xv) + gamma1
        plus_xe = xtv*deg_v_inv_sqrt
        plus_xe = self.n*scatter_sum(plus_xe[V], E, dim=0, dim_size=num_edges)*deg_e_inv_sqrt
        in_xe = ind_inte*self.aggregate(((1 + ind_inte) * xte),deg_e_inv_sqrt,deg_v_inv ,E,V,num_edges,num_nodes)
        out_xe = rate2*ind_bde*self.aggregate(ind_inte * xte ,deg_e_inv_sqrt,deg_v_inv,E,V,num_edges,num_nodes)
        xe = in_xe+out_xe+ind_inte*plus_xe+w2*ind_bde*plus_xe                                                 
        xe = self.fc2(xe) + gamma2
        xv = self.norm(xv)
        xe = self.norm(xe) 
        return xv,xe

    @staticmethod
    def aggregate(x, sqrt,inv,V, E, num_nodes=None, num_edges=None):
        x=x*sqrt
        edge_msg = scatter_sum(x[V], E, dim=0, dim_size=num_edges)  
        edge_msg = edge_msg * inv
        out = scatter_sum(edge_msg[E], V, dim=0, dim_size=num_nodes) 
        out = out * sqrt
        return out
    


class HGNNlayer1(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias=True, act='gelu', drop=0.3,tau = 0.1):
        super().__init__()
        self.tau = tau
        self.fc = FeedForwardLayer(in_dim, hid_dim, out_dim, bias, act, drop)
        self.node_lin = nn.Linear(in_dim, hid_dim, bias=bias)
        self.edge_lin = nn.Linear(in_dim, hid_dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x, V,E):
        num_nodes = x.size(0)
        num_edges = E.max().item() + 1
        deg_v = scatter_sum(torch.ones_like(V, dtype=x.dtype), V, dim=0, dim_size=num_nodes)
        deg_v_inv_sqrt = deg_v.pow(-0.5)
        deg_v_inv_sqrt[deg_v_inv_sqrt == float('inf')] = 0
        deg_e = scatter_sum(torch.ones_like(E, dtype=x.dtype), E, dim=0, dim_size=num_edges)
        deg_e_inv = deg_e.pow(-1)
        deg_e_inv[deg_e_inv == float('inf')] = 0
        x = x * deg_v_inv_sqrt.unsqueeze(1)
        x_edge = scatter_sum(x[V], E, dim=0, dim_size=num_edges)
        x_edge = x_edge * deg_e_inv.unsqueeze(1)
        x_node = scatter_sum(x_edge[E], V, dim=0, dim_size=num_nodes)
        x_node = x_node * deg_v_inv_sqrt.unsqueeze(1)
        x_node = x_node / x_node.std(dim=0, keepdim=True)
        out = self.fc(x_node)
        return out
    
class HGNNlayer2(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias=True, act='gelu', drop=0.3,tau = 0.1):
        super().__init__()
        self.tau = tau
        self.fc = FeedForwardLayer(in_dim, hid_dim, out_dim, bias, act, drop)
        self.edge_lin = nn.Linear(in_dim, hid_dim, bias=bias)
        self.node_lin = nn.Linear(in_dim, hid_dim, bias=bias)

    def forward(self, xe, V,E):
        num_nodes = V.max().item() + 1
        num_edges = E.max().item() + 1
        deg_v = scatter_sum(torch.ones_like(V, dtype=xe.dtype), V, dim=0, dim_size=num_nodes)
        deg_v_inv = deg_v.pow(-1)
        deg_v_inv[deg_v_inv == float('inf')] = 0
        deg_e = scatter_sum(torch.ones_like(E, dtype=xe.dtype), E, dim=0, dim_size=num_edges)
        deg_e_inv_sqrt = deg_e.pow(-0.5)
        deg_e_inv_sqrt[deg_e_inv_sqrt == float('inf')] = 0
        xe = xe * deg_e_inv_sqrt.unsqueeze(1)
        xv = scatter_sum(xe[E], V, dim=0, dim_size=num_nodes)
        xv = xv * deg_v_inv.unsqueeze(1)
        xe = scatter_sum(xv[V], E, dim=0, dim_size=num_edges)
        xe = xe * deg_e_inv_sqrt.unsqueeze(1)
        xe = xe / xe.std(dim=0, keepdim=True)
        out = self.fc(xe)
        return out
    