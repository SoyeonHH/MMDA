import os
import sys
import math
from pyexpat import model
import numpy as np
from random import random
import time
import datetime
import wandb
from transformers import BertTokenizer

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver, ConfidNet_Trainer
from inference import Inference
from utils.tools import *
from utils.eval_metrics import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.chdir(os.getcwd())
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from MISA.utils import to_gpu, to_cpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import MISA.models as models

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def main():

    # Setting training log
    args = get_config()
    wandb.init(project="MMDA")
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

    solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # try:
    model = load_model(train_config)
    # except:
    #     model = solver.train()

    tester = Inference(test_config, test_data_loader, model=model)
    tester.inference()
    
    if args.use_kt == True and args.kt_model == 'Dynamic-tcp':
        # Training the confidnet with zero_label_processed version
        train_data_loader_nonzero = get_loader(train_config, shuffle = True, zero_label_process=True)
        dev_data_loader_nonzero = get_loader(dev_config, shuffle = False, zero_label_process=True)
        test_data_loader_nonzero = get_loader(test_config, shuffle = False, zero_label_process=True)

        try:
            trained_confidnet = load_model(train_config, confidNet=True)
        except:
            confidnet_trainer = ConfidNet_Trainer(train_config, train_data_loader_nonzero, dev_data_loader_nonzero, test_data_loader_nonzero, model=model)
            confidnet_trainer.build()
            trained_confidnet = confidnet_trainer.train()
        
        solver_dkt_tcp = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, \
                                model=model, confidnet=trained_confidnet)
        solver_dkt_tcp.build()
        model = solver_dkt_tcp.train(additional_training=True)

        tester = Inference(test_config, test_data_loader, model=model, dkt=True)
        tester.inference()

if __name__ == "__main__":
    main()