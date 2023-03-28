import os
import sys
import math
from pyexpat import model
import numpy as np
from random import random
import wandb

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver
from solver_dkt_tcp import Solver_DKT_TCP
from solver_dkt_ce import Solver_DKT_CE
from inference import Inference
from utils.tools import *
from transformers import BertTokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import config
from utils.tools import *
from utils.eval_metrics import *
import time
import datetime
import wandb
import warnings

warnings.filterwarnings("ignore")

os.chdir(os.getcwd())
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, to_cpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models


class ConfidNet(Object):
    def __init__(self, config, dataloader, confidnet=None, checkpoint=None):
        self.config = config
        self.dataloader = dataloader
        self.confidnet = confidnet
        self.checkpoint = checkpoint
        self.device = torch.device(config.device)

        # TODO: Implement ConfidNet training module
        

def main():

    # Setting training log
    args = get_config()
    wandb.init(project="MISA-confidNet")
    wandb.config.update(args)

    # Setting random seed
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # Loading classification model
    model = getattr(models, train_config.model)(train_config)
    model.load_state_dict(load_model(train_config, name=train_config.model))
    model = model.to(train_config.device)


    


if __name__ == "__main__":
    main()