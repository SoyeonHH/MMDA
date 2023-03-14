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
#from solver_dkt_ce import Solver
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

os.chdir(os.getcwd())
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, to_cpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def main():

    # Setting training log
    args = get_config()
    wandb.init(project="multimodal_kt_dist")
    wandb.config.update(args)

    # Setting random seed
    random_name = str(random())
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

    # Solver is a wrapper for model traiing and testing
    if args.use_kt == False or args.kt_model == 'Static':
        solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)
    elif args.kt_model == 'Dynamic-ce':
        solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    #torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
    #torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
    solver.model.load_state_dict(torch.load('checkpoints/model_2023-03-13_19:18:26.std'))
    solver.optimizer.load_state_dict(torch.load('checkpoints/optim_2023-03-13_19:18:26.std'))

    # Test the model
    tester = Inference(test_config, test_data_loader, model=solver.model)
    tester.inference()

if __name__ == "__main__":
    main()