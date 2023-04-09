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
word_emb_path = '/home/iknow/workspace/multimodal/glove.840B.300d.txt'
# word_emb_path = '/data2/multimodal/glove.840B.300d.txt'
assert(word_emb_path is not None)


sdk_dir = Path('/home/iknow/workspace/multimodal/CMU-MultimodalSDK')
data_dir = Path('/home/iknow/workspace/multimodal/MOSEI')
# sdk_dir = Path('/data2/multimodal/CMU-MultimodalSDK')
# data_dir = Path('/data2/multimodal/MOSEI')
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
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)
                
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
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--eval_mode', type=str, default='micro', help='one of {micro, macro, weighted}')
    parser.add_argument('--checkpoint', type=str, default=None)

    # Bert
    parser.add_argument('--use_bert', type=str2bool, default=True)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)

    # Data
    parser.add_argument('--data', type=str, default='mosei')
    parser.add_argument('--aligned', type=str2bool, default=True)

    # Model
    parser.add_argument('--model', type=str, default='TAILOR', help='one of {Early, TFN, MISA, TAILOR}')

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=6)   # Fixed to classify
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--patience', type=int, default=6)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--extractor', type=str, default='lstm', help='one of {lstm, trasformer}')
    parser.add_argument('--rnncell', type=str, default='lstm')

    parser.add_argument('--embedding_size', type=int, default=300, help='text_feature_dimenstion')
    parser.add_argument('--video_dim', type=int, default=35, help='video_feature_dimenstion')
    parser.add_argument('--audio_dim', type=int, default=74, help='audio_feature_dimenstion')

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='leakyrelu')
    parser.add_argument('--threshold', type=float, default=0.35)

    # Train MISA
    parser.add_argument('--diff_weight', type=float, default=0.3)   # beta
    parser.add_argument('--sim_weight', type=float, default=0.7)    # alpha
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=0.7)  # gamma 

    # Train TAILOR
    parser.add_argument('--max_words', type=int, default=60, help='')
    parser.add_argument('--max_frames', type=int, default=60, help='')
    parser.add_argument('--max_sequence', type=int, default=60, help='')
    parser.add_argument('--max_label', type=int, default=6, help='')
    parser.add_argument("--bert_model", default="bert-base", type=str, required=False, help="Bert module")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--audio_model", default="audio-base", type=str, required=False, help="Audio module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.") 
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--bert_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=3, help="Layer No. of audio")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=1, help="Layer NO. of decoder.")

    # Train DKT
    parser.add_argument('--use_kt', type=str2bool, default=True)
    parser.add_argument('--kt_model', type=str, 
                    default='Dynamic-tcp', help='one of {Static, Dynamic-ce, Dynamic-tcp}')
    parser.add_argument('--kt_weight', type=float, default=10000.0)
    parser.add_argument('--n_epoch_dkt', type=int, default=30)
    parser.add_argument('--dynamic_method', type=str, default='ratio', help='one of {threshold, ratio, noise_level}')

    # Train ConfidNet
    parser.add_argument('--use_confidNet', type=str2bool, default=True)
    parser.add_argument('--n_epoch_conf', type=int, default=10)
    parser.add_argument('--conf_loss', type=str, default='mse', help='one of {mse, focal, ranking}')
    parser.add_argument('--conf_lr', type=float, default=1e-5)
    parser.add_argument('--conf_dropout', type=float, default=0.6)
    parser.add_argument('--conf_weight', type=float, default=0.3)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace to dict
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)    