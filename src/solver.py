import os
import sys
import math
from math import isnan
import re
import pickle
import gensim
from create_dataset import PAD
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from transformers import BertTokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import config
from utils.tools import *
from utils.eval import *
import time
import datetime
import wandb

import hypertune
hpt = hypertune.HyperTune()

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, to_cpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        
        if torch.cuda.is_available():
            self.device = torch.device(train_config.device)
        else:
            self.device = torch.device("cpu")
        print(f"current device: {self.device}")
    
    # @time_desc_decorator('Build Graph')
    def build(self, cuda=True): 
        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert:
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False
        
        # # Multi-GPU training setting
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)
        
        # if torch.cuda.is_available() and cuda:
        self.model.to(self.device)

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)

    # @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1
        
        best_valid_loss = float('inf')
        best_train_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        train_losses = []
        # total_start = time.time()
        for e in range(self.train_config.n_epoch):
            self.model.train()

            train_loss = []

            for idx, batch in enumerate(tqdm(self.train_data_loader)):
                self.model.zero_grad()
                _, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                label_input, label_mask = Solver.get_label_input()

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
                label_input, label_mask = to_gpu(label_input), to_gpu(label_mask)

                loss, y_tilde, predicted_labels, _ = self.model(t, v, a, l, \
                    bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, training=True)
                # y_tilde = y_tilde.squeeze()

                loss.backward()
                
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                self.optimizer.step()

                train_loss.append(loss.item())
            

            train_losses.append(train_loss)
            train_avg_loss = round(np.mean(train_loss), 4)
            print(f"Training loss: {train_avg_loss}")


            ##########################################
            # model evaluation with dev set
            ##########################################

            valid_loss, valid_acc, preds, truths = self.eval(mode="dev")

            print("-" * 100)
            print("Epochs: {}, Valid loss: {}, Valid acc: {}".format(e, valid_loss, valid_acc))
            print("-" * 100)


            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                best_results = preds
                best_truths = truths
                best_epoch = e
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                self.train_config.checkpoint = f'checkpoints/model_{self.train_config.name}.std'
                
                curr_patience = patience
                # 임의로 모델 경로 지정 및 저장
                save_model(self.train_config, self.model, name=self.train_config.model)
                # Print best model results
                eval_values_best = get_metrics(best_truths, best_results, average=self.train_config.eval_mode)
                print("-"*50)
                print("epoch: {}, valid_loss: {}, valid_acc: {}, f1: {}, precision: {}, recall: {}".format( \
                    best_epoch, valid_loss, eval_values_best['acc'], eval_values_best['f1'], eval_values_best['precision'], eval_values_best['recall']))
                # print("best results: ", best_results)
                # print("best truths: ", best_truths)
                print("-"*50)

            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            eval_values = get_metrics(truths, preds, average=self.train_config.eval_mode)

            wandb.log(
                (
                    {
                        "train_loss": train_avg_loss,
                        "valid_loss": valid_loss,
                        "test_f_score": eval_values['weighted_f1'],
                        "test_precision": eval_values['weighted_precision'],
                        "test_recall": eval_values['weighted_recall'],
                        "test_acc2": eval_values['acc'],
                    }
                )
            )

            # hyperparameter tuning report
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="accuracy",
                metric_value=eval_values['acc'],
                global_step=e
            )

        ##########################################
        # Test
        ##########################################

        print("model training is finished.")
        train_loss, acc, test_preds, test_truths = self.eval(mode="test", to_print=True)
        print('='*50)
        print(f'Best epoch: {best_epoch}')
        eval_values_best = get_metrics(best_truths, best_results, average=self.train_config.eval_mode)
        best_acc, best_f1, best_precision, best_recall = \
             eval_values_best['acc'], eval_values_best['f1'], eval_values_best['precision'], eval_values_best['recall']
        print(f'Accuracy: {best_acc}')
        print(f'F1 score: {best_f1}')
        print(f'Precision: {best_precision}')
        print(f'Recall: {best_recall}')
        # total_end = time.time()
        # total_duration = total_end - total_start
        # print(f"Total training time: {total_duration}s, {datetime.timedelta(seconds=total_duration)}")


    
    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_{self.train_config.name}.std'))
            

        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()

                _, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                label_input, label_mask = Solver.get_label_input()

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
                label_input = to_gpu(label_input)
                label_mask = to_gpu(label_mask)

                loss, predicted_scores, predicted_labels, hidden_state = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, \
                               masked_modality=None, training=False)

                eval_loss.append(loss.item())

                # y_tilde = torch.argmax(y_tilde, dim=1)
                # emo_label = torch.argmax(emo_label, dim=1)
                y_pred.append(predicted_labels.detach().cpu().numpy())
                y_true.append(emo_label.detach().cpu().numpy())


        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()   # (1871, 6)
        y_pred = np.concatenate(y_pred, axis=0).squeeze()   # (1871, 6)

        accuracy = get_accuracy(y_true, y_pred)

        return eval_loss, accuracy, y_pred, y_true


    @staticmethod
    def get_label_input():
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask
