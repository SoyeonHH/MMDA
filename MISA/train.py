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
from confidNet import ConfidNet_Trainer
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

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def main():

    # Setting training log
    args = get_config()
    wandb.init(project="MISA-classification")
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
    # if args.use_kt == True and args.kt_model == 'Dynamic-tcp':
    #     solver = Solver_DKT_TCP(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)
    # elif args.use_kt == True and args.kt_model == 'Dynamic-ce':
    #     solver = Solver_DKT_CE(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)
    # else:
    solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    try:
        pre_trained_model = load_model(args, name=args.model)
    except:
        pre_trained_model = solver.train()
    
    if args.use_kt == True:
        if args.kt_model == 'Dynamic-tcp':
            # Training the confidnet with zero_label_processed version
            train_data_loader = get_loader(train_config, shuffle = True, zero_label_process=True)
            dev_data_loader = get_loader(dev_config, shuffle = False, zero_label_process=True)
            test_data_loader = get_loader(test_config, shuffle = False, zero_label_process=True)

            confidnet_trainer = ConfidNet_Trainer(train_config, train_data_loader, dev_data_loader, test_data_loader)
            trained_confidnet = confidnet_trainer.train()
            
        dkt_solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=pre_trained_model)
        dkt_solver.train_DKT(confidnet=trained_confidnet)

    tester = Inference(test_config, test_data_loader, model=solver.model)
    tester.inference()

if __name__ == "__main__":
    main()