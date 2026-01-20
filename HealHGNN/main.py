import torch
import os
import argparse
from node_classification import Exp
from logger import create_logger
from utils.config import load_config, save_config
from utils.eval_utils import set_seed
import json

set_seed(2025)

parser = argparse.ArgumentParser(description='')

# Experiment settings
parser.add_argument('--task', type=str, default='NC',
                    choices=['NC'])
parser.add_argument('--dataset', type=str, default='senate-committees', help="data name")
parser.add_argument('--val_every', type=int, default=5)
parser.add_argument('--exp_iters', type=int, default=10)
parser.add_argument('--log_dir', type=str, default="./logs/")
parser.add_argument('--result_dir', type=str, default="./results/")
parser.add_argument('--task_model_path', type=str)  # necessary
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--data_dir', type=str,default= './datasets')
parser.add_argument('--raw_data_dir', type=str,default= './raw_data')
parser.add_argument('--method', default='HEAL', help='model type')

# Base Params
parser.add_argument('--add_self_loop', action='store_true', help='add self loop to adjacency')
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
parser.add_argument('--embed_dim', type=int, default=512, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--input_act', type=str, default='gelu', help='activation function for input layer')
parser.add_argument('--norm', type=str, default='ln', help='Normalization of Batch Norm or Layer Norm')
parser.add_argument('--bias', action='store_false', help='use bias for linear layer')
parser.add_argument('--tau', type=float, default=0.5, help='temperature for sigmoid')
parser.add_argument('--act', type=str, default='gelu', help='activation function')
parser.add_argument('--layer_wise',action='store_false', help='whether to use layer-wise training')
parser.add_argument('--u', type=float, default=0.5,
                    help='Weight for hyperedge-to-node message passing')
parser.add_argument('--lam', type=float, default=0.5,
                    help='Weight for node-to-hyperedge message passing')

# Dataset specific arguments
parser.add_argument('--train_prop', type=float, default=0.5)
parser.add_argument('--valid_prop', type=float, default=0.25)
parser.add_argument('--feature_noise', default='1', type=str, help='std for synthetic feature noise')
parser.add_argument('--normtype', default='all_one', choices=['all_one','deg_half_sym'])
parser.add_argument('--exclude_self', action='store_true', help='whether the he contain self node or not')

# Node Classification
parser.add_argument('--lr_nc', type=float, default=3e-5)
parser.add_argument('--weight_decay_nc', type=float, default=5e-4)
parser.add_argument('--epochs_nc', type=int, default=1000)
parser.add_argument('--patience_nc', type=int, default=25)

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

args= parser.parse_args()

json_dir = f"./configs/{args.task}"
json_path = f"{json_dir}/{args.dataset}.json"
if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)

if os.path.exists(json_path):
    print(f"Loading config file: {json_path}")
    with open(json_path, 'r') as f:
        json_config = json.load(f)
else:
    print(f"Saving config file: {json_path}")
    with open(json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    json_config = {}

for k, v in json_config.items():
    if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
        setattr(args, k, v)
configs = args
if not os.path.exists(configs.log_dir):
    os.makedirs(configs.log_dir, exist_ok=True)
log_path = (f"{configs.log_dir}/{configs.task}_{configs.dataset}"
            f"_NL{configs.n_layers}_HD{configs.hid_dim}_DP{configs.dropout}_ACT{configs.act}"
            f"_NORM{configs.norm}_B{configs.bias}_LRNC{configs.lr_nc}.log")
configs.log_path = log_path

if not os.path.exists(configs.result_dir):
    os.makedirs(configs.result_dir, exist_ok=True)
result_path = f"{configs.result_dir}/{configs.task}_{configs.dataset}.txt"
configs.result_path = result_path

if configs.task_model_path is None:
    configs.task_model_path = f"{configs.task}_{configs.dataset}_model.pt"

print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

if configs.task == "NC":
    exp = Exp(configs)
else:
    raise NotImplementedError
exp.train()
torch.cuda.empty_cache()