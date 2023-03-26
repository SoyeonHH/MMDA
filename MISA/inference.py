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
import json
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from create_dataset import PAD
from solver import *
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from data_loader import get_loader
from solver import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import config
from utils.tools import *
from utils.eval import *
from utils.functions import *
import time
import datetime
import wandb
import csv
import warnings

warnings.filterwarnings("ignore")

os.chdir(os.getcwd())
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
            if checkpoint:
                self.model.load_state_dict(torch.load(checkpoint))
            else:
                self.model.load_state_dict(load_model(config, name=config.model))
        
        self.model = self.model.to(self.device)

        if self.confidence_model is None and config.use_confidNet:
            self.confidence_model = getattr(models, "ConfidenceRegressionNetwork")(self.config, self.config.hidden_size*6, \
                num_classes=1, dropout=self.config.conf_dropout)
            self.confidence_model.load_state_dict(load_model(config, name=self.config.model, confidNet=True))

            self.confidence_model = self.confidence_model.to(self.device)
        
        
    
    def inference(self):
        print("Start inference...")
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss = []
        results = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                self.model.zero_grad()
                result = dict()

                actual_words, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
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
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None,\
                        dynamic_weights=None, training=False)
                
                eval_loss.append(loss.item())
                y_true.append(emo_label.cpu().numpy())
                y_pred.append(predicted_labels.cpu().numpy())

                #result["id"] = ids
                #result["input_sentence"] = actual_words
                #result["label"] = emo_label.cpu().numpy()
                #result["prediction"] = predicted_labels.cpu().numpy()
                # result["id"] = ids[0]
                # result["input_sentence"] = actual_words[0]
                # result["label"] = emo_label.cpu().numpy()[0]
                # result["prediction"] = predicted_labels.cpu().numpy()[0]
                # result["Original_Loss"] = hidden_state[0][0].item()
                # result["T_Masked_Loss"] = hidden_state[0][1].item()
                # result["V_Masked_Loss"] = hidden_state[0][3].item()
                # result["A_Masked_Loss"] = hidden_state[0][2].item()
                
                results.append(result)

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        # columns = ["id", "input_sentence", "label", "prediction", "Original_Loss", "T_Masked_Loss", "V_Masked_Loss","A_Masked_Loss"]
        # if self.config.use_kt:
        #     file_name = "/results_{}_kt-{}({})-dropout({})-batchsize({}).csv".format(\
        #         self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size)
        # else:
        #     file_name = "/results_{}_dropout({})-batchsize({}).csv".format(self.config.model, self.config.dropout, self.config.batch_size)
        
        # with open(os.getcwd() + file_name, "w") as f:
        #     writer = csv.DictWriter(f, fieldnames=columns)
        #     writer.writeheader()
        #     writer.writerows(results)
        
        # Total results log
        accuracy = get_accuracy(y_true, y_pred)
        eval_values = get_metrics(y_true, y_pred, average=self.config.eval_mode)

        print("="*50)
        print("Loss: {:.4f}, Accuracy: {:.4f}".format(eval_loss, accuracy))
        print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_values['precision'], eval_values['recall'], eval_values['f1']))
        print("="*50)

        # Save metric results into json
        total_results = {
            "loss": eval_loss,
            "accuracy": accuracy,
            "precision": eval_values['precision'],
            "recall": eval_values['recall'],
            "f1": eval_values['f1']
        }

        if self.config.use_kt:
            json_name = "/results_{}_kt-{}({})-dropout({})-batchsize({}).json".format(\
                self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size)
        else:
            json_name = "/results_{}_dropout({})-batchsize({}).json".format(self.config.model, self.config.dropout, self.config.batch_size)
        
        with open(os.getcwd() + json_name, "w") as f:
            json.dump(total_results, f, indent=4)


    
    def inference_with_confidnet(self):
        print("Start inference...")
        self.loss_mcp = nn.CrossEntropyLoss(reduction="mean")
        self.loss_tcp = nn.MSELoss(reduction="mean")

        self.model.eval()
        self.confidence_model.eval()

        y_true, y_pred = [], []
        tcp_true, tcp_pred = [], []
        eval_loss, eval_conf_loss = [], []

        results = defaultdict(list)

        with torch.no_grad():

            for batch in tqdm(self.dataloader):
                self.model.zero_grad()
                self.confidence_model.zero_grad()

                actual_words, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
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

                # Mutli-Modal Fusion Model
                loss, predicted_scores, predicted_labels, hidden_state = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None,\
                        dynamic_weights=None, training=False)
                
                # Text Masking Fusion Model
                loss_t, predicted_scores_t, predicted_labels_t, hidden_state_t = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality="text",\
                        dynamic_weights=None, training=False)

                # Video Masking Fusion Model
                loss_v, predicted_scores_v, predicted_labels_v, hidden_state_v = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality="video",\
                        dynamic_weights=None, training=False)
                
                # Audio Masking Fusion Model
                loss_a, predicted_scores_a, predicted_labels_a, hidden_state_a = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask,labels=emo_label, masked_modality="audio",\
                        dynamic_weights=None, training=False)
                
                
                emo_label = emo_label.type(torch.float)

                # Confidence Model
                predicted_tcp, _ = self.confidence_model(hidden_state)
                predicted_tcp_t, _ = self.confidence_model(hidden_state_t)
                predicted_tcp_v, _ = self.confidence_model(hidden_state_v)
                predicted_tcp_a, _ = self.confidence_model(hidden_state_a)

                predicted_tcp, predicted_tcp_t, predicted_tcp_v, predicted_tcp_a = \
                    predicted_tcp.squeeze(), predicted_tcp_t.squeeze(), predicted_tcp_v.squeeze(), predicted_tcp_a.squeeze()
                
                
                # Calculate target tcp
                target_tcp = get_tcp_target(emo_label, predicted_scores)

                # Calculate loss
                conf_loss = self.get_conf_loss(predicted_scores, emo_label, predicted_tcp)

                # Make result dictionary
                results["id"].extend(ids)
                results["input_sentence"].extend(actual_words)
                results["label"].extend(emo_label.detach().cpu().numpy())
                results["prediction"].extend(predicted_labels.detach().cpu().numpy())
                results["target_tcp"].extend(target_tcp.detach().cpu().numpy())
                results["pred_tcp"].extend(predicted_tcp.detach().cpu().numpy())
                results["pred_tcp-t"].extend(predicted_tcp_t.detach().cpu().numpy())
                results["pred_tcp-v"].extend(predicted_tcp_v.detach().cpu().numpy())
                results["pred_tcp-a"].extend(predicted_tcp_a.detach().cpu().numpy())

                eval_loss.append(loss.item())
                eval_conf_loss.append(conf_loss.item())

                y_pred.append(predicted_labels.detach().cpu().numpy())
                y_true.append(emo_label.detach().cpu().numpy())
                # tcp_pred.append(predicted_tcp.detach().cpu().numpy())
                # tcp_true.append(target_tcp.detach().cpu().numpy())

            eval_loss = np.mean(eval_loss)
            eval_conf_loss = np.mean(eval_conf_loss)
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()


        # columns = ["id", "input_sentence", "label", "prediction", "tcp", "pred_tcp", "pred_tcp-t", "pred_tcp-v", "pred_tcp-a"]
        file_name = "/results_kt-{}({})-dropout({})-confidNet-dropout({})-batchsize({}).csv".format(\
            self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.conf_dropout, self.config.batch_size)
        
        with open(os.getcwd() + file_name, 'w') as f:
            key_list = list(results.keys())
            writer = csv.writer(f)
            writer.writerow(results.keys())
            for i in range(len(key_list)):
                writer.writerow([results[x][i] for x in key_list])
        
        # Calculate accuracy
        accuracy = get_accuracy(y_true, y_pred)
        eval_values = get_metrics(y_true, y_pred, average=self.config.eval_mode)

        print("="*50)
        print("Loss: {:.4f}, Conf Loss: {:.4f}, Accuracy: {:.4f}".format(eval_loss, eval_conf_loss, accuracy))
        print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_values['precision'], eval_values['recall'], eval_values['f1']))
        print("="*50)

        # Save metric results into json
        total_results = {
            "loss": eval_loss,
            "conf_loss": eval_conf_loss,
            "accuracy": accuracy,
            "precision": eval_values['precision'],
            "recall": eval_values['recall']
        }

        json_name = "/results_kt-{}({})-dropout({})-confidNet-dropout({})-batchsize({}).json".format(\
        self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.conf_dropout, self.config.batch_size)

        with open(os.getcwd() + json_name, 'w') as f:
            json.dump(total_results, f, indent=4)



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

        mcp_loss = self.loss_mcp(pred, truth) * self.config.mcp_weight

        if self.config.use_mcp:
            return torch.add(tcp_loss, mcp_loss, alpha=self.train_config.mcp_weight)
        else:
            return tcp_loss


def main():

    # Setting testing log
    args = config.get_config()

    test_config = config.get_config(mode='test')
    # test_config.batch_size = 1
    test_data_loader = get_loader(test_config, shuffle=False)

    tester = Inference(test_config, test_data_loader, checkpoint=args.checkpoint)

    if test_config.use_confidNet:
        tester.inference_with_confidnet()
    else:
        tester.inference()


if __name__ == "__main__":
    main()