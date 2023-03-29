import os
import sys
import math
from pyexpat import model
import numpy as np
from random import random
import wandb
from tqdm import tqdm

from config import get_config, activation_dict
from data_loader import get_loader
from utils.tools import *
from transformers import BertTokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utils.tools import *
from utils.eval_metrics import *
from utils.functions import *
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


class ConfidNet_Trainer(object):
    def __init__(self, config, train_data_loader, dev_data_loader, test_data_loader, model=None, confidnet=None):
        self.config = config
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.confidnet = confidnet
        self.model = model
        self.device = torch.device(config.device)
        print(f"current device: {self.device}")
    
    def build(self):
        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)
            self.model.load_state_dict(load_model(self.config, name=self.config.model))
            self.model = self.model.to(self.device)

        self.model.eval()

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize confidence network
        if self.confidnet is None:
            self.confidnet = ConfidenceRegressionNetwork(self.config, input_dims=self.config.hidden_size*6, num_classes=1, dropout=self.config.conf_dropout)
            self.confidnet = self.confidnet.to(self.device)

        self.optimizer = torch.optim.Adam(self.confidnet.parameters(), lr=self.config.conf_lr)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.config.use_bert:
            if self.config.pretrained_emb is not None:
                self.model.embed.weight.data = self.config.pretrained_emb
            self.model.embed.requires_grad = False
        

    def train(self):
        self.build()
        self.confidnet.train()
        train_results = []
        best_valid_loss = float('inf')

        for epoch in range(self.config.n_epoch_conf):
            train_losses = []

            for i, batch in enumerate(tqdm(self.train_data_loader)):
                self.optimizer.zero_grad()

                actual_words, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                # batch_size = t.size(0)
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                emo_label = to_gpu(emo_label)
                # l = to_gpu(l)
                l = to_cpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                # Get the output from the classification model
                _, outputs, output_labels, hidden_state = self.model(t, v, a, l, \
                    bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, training=False)
                target_tcp = get_tcp_target(emo_label, outputs)
                
                loss, predicts = self.confidnet(hidden_state, target_tcp)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

                for idx in range(len(ids)):
                    train_result = {
                        "epoch": epoch,
                        "id": ids[idx],
                        "confid_loss": loss.item(),
                        "target_tcp": target_tcp[idx].item(),
                        "predict_tcp": predicts[idx].item(),
                        "emo_label": emo_label[idx].detach().cpu().numpy(),
                        "predict": outputs[idx].detach().cpu().numpy(),
                        "input_text": actual_words[idx]
                    }
                    train_results.append(train_result)
            
            train_avg_loss = np.mean(train_losses)
            print(f"Epoch: {epoch}, Train Loss: {train_avg_loss}")
        
            # Model Validation
            valid_loss, _ = self.eval(mode="dev")

            print("-" * 100)
            print("Epochs: {}, Valid loss: {}".format(epoch, valid_loss))
            print("-" * 100)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                print("Saving the best model...")
                save_model(self.config, self.confidnet, name=self.config.model, confidNet=True)

            wandb.log({"train_loss": train_avg_loss, "valid_loss": valid_loss})


        # Model Test
        print("Testing the model...")
        test_loss, test_result = self.eval(mode="test")
        print('='*50)
        print(f'Best epoch: {best_epoch}')
        print(f'Best valid loss: {best_valid_loss}')
        print(f'Test loss: {test_loss}')
        print('='*50)

        # Save the results
        save_results(self.config, train_results, mode="train")
        save_results(self.config, test_result, mode="test")

    
    def eval(self, mode=None):
        self.confidnet.eval()
        eval_losses = []
        eval_results = []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader
        
        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                self.confidnet.zero_grad()

                actual_words, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                emo_label = to_gpu(emo_label)
                # l = to_gpu(l)
                l = to_cpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                _, outputs, output_labels, hidden_state = self.model(t, v, a, l, \
                    bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, training=False)
                target_tcp = get_tcp_target(emo_label, outputs)
                
                loss, predicts = self.confidnet(hidden_state, target_tcp)

                eval_losses.append(loss.item())

                if mode == "test":
                    for idx in range(len(ids)):
                        eval_result = {
                            "id": ids[idx],
                            "confid_loss": loss.item(),
                            "target_tcp": target_tcp[idx].item(),
                            "predict_tcp": predicts[idx].item(),
                            "emo_label": emo_label[idx].detach().cpu().numpy(),
                            "predict": outputs[idx].detach().cpu().numpy(),
                            "input_text": actual_words[idx]
                        }
                        eval_results.append(eval_result)

        eval_avg_loss = np.mean(eval_losses)

        return eval_avg_loss, eval_results



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

    confidence_trainer = ConfidNet_Trainer(train_config, train_data_loader, dev_data_loader, test_data_loader)
    confidence_trainer.train()


if __name__ == "__main__":
    main()