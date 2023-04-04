import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# path to a pretrained word embedding file
# word_emb_path = '/home/iknow/workspace/multimodal/glove.840B.300d.txt'
word_emb_path = '/data2/multimodal/glove.840B.300d.txt'
assert(word_emb_path is not None)


# sdk_dir = Path('/home/iknow/workspace/multimodal/CMU-MultimodalSDK')
# data_dir = Path('/home/iknow/workspace/multimodal')
sdk_dir = Path('/data2/multimodal/CMU-MultimodalSDK')
data_dir = Path('/data2/multimodal/MOSEI')
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

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
        self.dataset_dir = self.data_dir = data_dir
        self.sdk_dir = sdk_dir

        # Glove path
        self.word_emb_path = word_emb_path

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
    parser.add_argument('--mode', type=str, default='train', help='one of train, dev or test')
    parser.add_argument()