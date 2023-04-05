import numpy as np
import wandb
from tqdm import tqdm

from config import get_config, activation_dict
from data_loader import get_loader
from transformers import BertTokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from utils.eval import *
from MISA.utils.functions import *


class ConfidenceRegressionNetwork(nn.Module):
    def __init__(self, config, input_dims, num_classes=1, dropout=0.1):
        super(ConfidenceRegressionNetwork, self).__init__()
        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes))
    
        self.sigmoid = nn.Sigmoid()
        
        self.loss_tcp = nn.MSELoss(reduction='mean')
    
    def forward(self, seq_input, targets):
        output = self.mlp(seq_input)
        output = self.sigmoid(output)
        loss = self.loss_tcp(output, targets)

        return loss, output
