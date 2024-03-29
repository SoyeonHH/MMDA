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

        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        self.criterion = criterion = nn.BCELoss(reduction="mean")
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        # self.loss_recon = MSE()
        self.loss_recon = nn.MSELoss(reduction="mean")
        self.loss_cmd = CMD()

        # Confidence regression loss
        self.loss_mcp = nn.CrossEntropyLoss(reduction="mean")
        self.loss_tcp = nn.MSELoss(reduction="mean")
        
        best_valid_loss = float('inf')
        best_train_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        train_losses = []
        # total_start = time.time()
        for e in range(self.train_config.n_epoch):
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_loss_conf = []
            train_loss = []

            # For label decoder inputs
            # label_input, label_mask = self.train_data_loader.dataset.get_label_input()

            for idx, batch in enumerate(tqdm(self.train_data_loader)):
                self.model.zero_grad()
                t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

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

                predicted_scores, predicted_labels = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                # y_tilde = y_tilde.squeeze()

                
                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                emo_label = emo_label.type(torch.float)

                cls_loss = self.get_cls_loss(predicted_scores, emo_label)
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                conf_loss = self.get_conf_loss(predicted_scores, emo_label)
                
                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss
                
                loss = cls_loss + \
                    self.train_config.diff_weight * diff_loss + \
                    self.train_config.sim_weight * similarity_loss + \
                    self.train_config.recon_weight * recon_loss

                if self.train_config.use_confidNet:
                    loss += self.train_config.conf_weight * conf_loss

                loss.backward()
                
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                self.optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_recon.append(recon_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())
                train_loss_conf.append(conf_loss.item())
                

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
                curr_patience = patience
                # 임의로 모델 경로 지정 및 저장
                save_model(self.train_config, self.model, self.train_config.data)
                # Print best model results
                eval_values = get_metrics(best_truths, best_results)
                print("-"*50)
                print("epoch: {}, valid_loss: {}, valid_acc: {}, f1: {}, precision: {}, recall: {}".format( \
                    best_epoch, valid_loss, eval_values['acc'], eval_values['f1'], eval_values['precision'], eval_values['recall']))
                # print("best results: ", best_results)
                # print("best truths: ", best_truths)
                print("-"*50)

            # else:
            #     curr_patience -= 1
            #     if curr_patience <= -1:
            #         print("Running out of patience, loading previous best model.")
            #         num_trials -= 1
            #         curr_patience = patience
            #         self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
            #         self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
            #         lr_scheduler.step()
            #         print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if self.train_config.eval_mode == "macro":
                wandb.log(
                    (
                        {
                            "train_loss": train_avg_loss,
                            "valid_loss": valid_loss,
                            "test_f_score": eval_values['f1'],
                            "test_precision": eval_values['precision'],
                            "test_recall": eval_values['recall'],
                            "test_acc2": eval_values['acc']
                        }
                    )
                )
            elif self.train_config.eval_mode == "micro":
                wandb.log(
                    (
                        {
                            "train_loss": train_avg_loss,
                            "valid_loss": valid_loss,
                            "test_f_score": eval_values['micro_f1'],
                            "test_precision": eval_values['micro_precision'],
                            "test_recall": eval_values['micro_recall'],
                            "test_acc2": eval_values['acc']
                        }
                    )
                )
            elif self.train_config.eval_mode == "weighted":
                wandb.log(
                    (
                        {
                            "train_loss": train_avg_loss,
                            "valid_loss": valid_loss,
                            "test_f_score": eval_values['weighted_f1'],
                            "test_precision": eval_values['weighted_precision'],
                            "test_recall": eval_values['weighted_recall'],
                            "test_acc2": eval_values['acc']
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

        train_loss, acc, test_preds, test_truths = self.eval(mode="test", to_print=True)
        print('='*50)
        print(f'Best epoch: {best_epoch}')
        eval_values_best = get_metrics(best_truths, best_results)
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
                t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

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

                predicted_scores, predicted_labels = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                # y_tilde = y_tilde.squeeze()

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                
                emo_label = emo_label.type(torch.float)
                
                cls_loss = self.get_cls_loss(predicted_scores, emo_label)
                loss = cls_loss

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

    
    def get_cls_loss(self, predicted_scores, emo_label):
        if self.train_config.data == "ur_funny":
            emo_label = emo_label.squeeze()
        
        emo_label = emo_label.type(torch.float)

        predicted_scores, emo_label = torch.permute(predicted_scores, (1, 0)), torch.permute(emo_label, (1, 0))

        cls_loss = 0.0
        for i in range(emo_label.size(0)):
            cls_loss += self.criterion(predicted_scores[i], emo_label[i])

        return cls_loss


    def get_domain_loss(self,):

        if self.train_config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self,):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss/3.0

        return loss

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/3.0
        return loss

    def get_conf_loss(self, pred, truth):
        tcp_loss = 0.0
        mcp_loss = 0.0

        pred, truth = torch.permute(pred, (1, 0)), torch.permute(truth, (1, 0))
        tcp = torch.permute(self.model.tcp, (1, 0))

        for i in range(truth.size(0)):
            tcp_loss += torch.div(self.loss_tcp(tcp[i], (truth[i] * pred[i])), torch.count_nonzero(truth[i]))
            mcp_loss += torch.div(self.loss_mcp(pred[i], truth[i]), torch.count_nonzero(truth[i]))

        return tcp_loss + mcp_loss