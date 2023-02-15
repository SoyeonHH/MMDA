# TODO: Implement inference.py
## Source: input dataset, dataloader, model, config, checkpoint
## Output: prediction, confidence, modality-masking confidence scores
## functions:
## 1. load dataset - id, input_sentence, label
## 2. load model - model, config, checkpoint
## 3. load dataloader - dataloader
## 4. return id, input_sentence, label, prediction, confidence, modality-masking confidence
## 5. save them as csv file

import os
import sys
import math
from math import isnan
import re
import pickle
import gensim
from collections import defaultdict

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

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, to_cpu, time_desc_decorator
import models

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Inference(object):
    def __init__(self, config, dataloader, model=None, confidence_model=None, checkpoint=None):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        self.confidence_model = confidence_model
        self.checkpoint = checkpoint
        self.device = torch.device(config.device)

        if self.model is None:
            self.model = getattr(models, config.model)(config)
            self.model = load_model(config, name=config.model)

        if self.confidence_model is None:
            self.confidence_model = getattr(models, "ConfidenceRegressionNetwork")(self.config.hidden_size*6, \
                num_classes=1, dropout=self.config.conf_dropout)
            self.confidence_model = load_model(config, name="confidNet")
        
        self.model = self.model.to(self.device)
        self.confidence_model = self.confidence_model.to(self.device)
    
    def inference(self):
        print("Start inference...")
        self.creterion = criterion = nn.BCELoss(reduction="mean")
        self.loss_mcp = nn.CrossEntropyLoss(reduction="mean")
        self.loss_tcp = nn.MSELoss(reduction="mean")

        self.model.eval()
        self.confidence_model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_conf_loss = [], []

        # TODO: use defaultdict

        confidence = []
        text_masking_conf, video_masking_conf, audio_masking_conf = [], [], []
        id, input_sentence, label = [], [], []

        with torch.no_grad():

            for batch in self.dataloader:
                self.model.zero_grad()
                self.confidence_model.zero_grad()

                t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                label_input, label_mask = self.get_label_input()

                

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

                # Mutli-Modal Fusion Model
                predicted_scores, predicted_labels, hidden_state = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, \
                        label_input, label_mask, groundTruth_labels=emo_label, training=False, masked_modality=None)
                
                # Text Masking Fusion Model
                predicted_scores_t, predicted_labels_t, hidden_state_t = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, \
                        label_input, label_mask, groundTruth_labels=emo_label, training=False, masked_modality="text")

                # Video Masking Fusion Model
                predicted_scores_v, predicted_labels_v, hidden_state_v = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, \
                        label_input, label_mask, groundTruth_labels=emo_label, training=False, masked_modality="video")
                
                # Audio Masking Fusion Model
                predicted_scores_a, predicted_labels_a, hidden_state_a = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, \
                        label_input, label_mask, groundTruth_labels=emo_label, training=False, masked_modality="audio")
                
                
                if self.config.data == "ur_funny":
                    y = y.squeeze()
                
                emo_label = emo_label.type(torch.float)

                # Confidence Model
                predicted_tcp = self.confidence_model(hidden_state)
                predicted_tcp_t = self.confidence_model(hidden_state_t)
                predicted_tcp_v = self.confidence_model(hidden_state_v)
                predicted_tcp_a = self.confidence_model(hidden_state_a)

                predicted_tcp, predicted_tcp_t, predicted_tcp_v, predicted_tcp_a = \
                    predicted_tcp.squeeze(), predicted_tcp_t.squeeze(), predicted_tcp_v.squeeze(), predicted_tcp_a.squeeze()

                cls_loss = self.get_cls_loss(predicted_scores, emo_label)
                loss = cls_loss

                conf_loss = self.get_conf_loss(predicted_scores, emo_label, predicted_tcp)

                eval_loss.append(loss.item())
                eval_conf_loss.append(conf_loss.item())

                y_pred.append(predicted_labels.detach().cpu().numpy())
                y_true.append(emo_label.detach().cpu().numpy())
                confidence.append(predicted_tcp.detach().cpu().numpy())
        
        eval_loss = np.mean(eval_loss)
        eval_conf_loss = np.mean(eval_conf_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        confidence = np.concatenate(confidence, axis=0).squeeze()

        accuracy = get_accuracy(y_true, y_pred)

        print("="*50)
        print("Loss: {:.4f}, Conf Loss: {:.4f}, Accuracy: {:.4f}".format(eval_loss, eval_conf_loss, accuracy))
        eval_values = get_metrics(y_true, y_pred)
        print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_values['precision'], eval_values['recall'], eval_values['f1']))
        print("="*50)


    def get_cls_loss(self, predicted_scores, emo_label):
        if self.train_config.data == "ur_funny":
            emo_label = emo_label.squeeze()
        
        emo_label = emo_label.type(torch.float)

        predicted_scores, emo_label = torch.permute(predicted_scores, (1, 0)), torch.permute(emo_label, (1, 0)) # (num_classes, batch_size)

        cls_loss = 0.0

        # summation of loss for each label
        for i in range(emo_label.size(0)):
            cls_loss += self.criterion(predicted_scores[i], emo_label[i])

        return cls_loss

    def get_conf_loss(self, pred, truth, predicted_tcp):    # pred: (batch_size, num_classes), truth: (batch_size, num_classes)
        tcp_loss = 0.0
        mcp_loss = 0.0
        tcp_batch = []

        for i in range(truth.size(0)):  # for each batch
            tcp = 0.0
            for j in range(truth[i].size(0)):   # for each class
                tcp += pred[i][j] * truth[i][j]
            tcp = tcp / torch.count_nonzero(truth[i]) if torch.count_nonzero(truth[i]) != 0 else 0.0
            tcp_batch.append(tcp)
        
        tcp_batch = to_gpu(torch.tensor(tcp_batch))
        tcp_loss = self.loss_tcp(predicted_tcp, tcp_batch)

        # pred, truth = torch.permute(pred, (1, 0)), torch.permute(truth, (1, 0)) # (num_classes, batch_size)

        # for i in range(truth.size(0)):
        #     mcp_loss += self.loss_mcp(pred[i], truth[i])
        # mcp_loss = mcp_loss / truth.size(0)

        return tcp_loss