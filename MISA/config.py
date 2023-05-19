import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# path to a pretrained word embedding file
# word_emb_path = '/home/iknow/workspace/multimodal/glove.840B.300d.txt'
word_emb_path = '/data2/multimodal/glove.840B.300d.txt'
assert(word_emb_path is not None)


# sdk_dir = Path('/home/iknow/workspace/multimodal/CMU-MultimodalSDK')
# data_dir = Path('/home/iknow/workspace/multimodal')
sdk_dir = Path('/data2/multimodal/CMU-MultimodalSDK')
data_dir = Path('/data2/multimodal')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath('MOSEI')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}

mosi_hp = {
    'activate': 'relu',
    'batch_size': 64,
    'alpha': 1.0,
    'beta': 0.3,
    'gamma': 1.0,
    'dropout': 0.5
}

mosei_hp = {
    'activate': 'leakyrelu',
    'batch_size': 16,
    'alpha': 0.7,
    'beta': 0.3,
    'gamma': 0.7,
    'dropout': 0.1,
    'text_dim': 300,
    'video_dim': 35,
    'audio_dim': 74
}

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]
        self.sdk_dir = sdk_dir
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        # self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--eval_mode', type=str, default='micro', help='one of {micro, macro, weighted}')
    parser.add_argument('--freeze', type=str2bool, default=False)
    parser.add_argument('--checkpoint', type=str, default=None)

    # Bert
    parser.add_argument('--use_bert', type=str2bool, default=True)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)

    # Data
    parser.add_argument('--data', type=str, default='mosei')

    # _args = parser.parse_args()
    # dataset_default_hp = mosi_hp if _args.data.strip() == 'mosi' else mosei_hp

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=6)   # Fixed to classify
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--n_epoch_conf', type=int, default=10)
    parser.add_argument('--n_epoch_dkt', type=int, default=30)
    parser.add_argument('--patience', type=int, default=6)

    parser.add_argument('--diff_weight', type=float, default=0.3)   # beta
    parser.add_argument('--sim_weight', type=float, default=0.7)    # alpha
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=0.7)  # gamma
    parser.add_argument('--conf_weight', type=float, default=0.3)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") 

    parser.add_argument('--extractor', type=str, default='lstm', help='one of {lstm, trasformer}')
    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='leakyrelu')
    parser.add_argument('--threshold', type=float, default=0.35)

    # Model
    parser.add_argument('--model', type=str,
                        default='MISA', help='one of {MISA, TFN, Early}')
    parser.add_argument('--use_confidNet', type=str2bool, default=True)
    parser.add_argument('--conf_loss', type=str, default='mse', help='one of {mse, focal, ranking}')
    parser.add_argument('--conf_lr', type=float, default=1e-5)
    parser.add_argument('--conf_dropout', type=float, default=0.6)
    parser.add_argument('--use_mcp', type=str2bool, default=False)
    parser.add_argument('--mcp_weight', type=float, default=0.1)
    
    parser.add_argument('--use_kt', type=str2bool, default=True)
    parser.add_argument('--kt_model', type=str, 
                        default='Dynamic-tcp', help='one of {Static, Dynamic-ce, Dynamic-tcp}')
    parser.add_argument('--kt_weight', type=float, default=10000.0)
    parser.add_argument('--dynamic_method', type=str, default='ratio', help='one of {threshold, ratio}')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
