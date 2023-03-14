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
            self.confidence_model.load_state_dict(load_model(config, name="confidNet"))

            self.confidence_model = self.confidence_model.to(self.device)
        
        
    
    def inference(self):
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
                        text_weight=None, video_weight=None, audio_weight=None, training=False)
                
                eval_loss.append(loss.item())
                y_true.append(emo_label.cpu().numpy())
                y_pred.append(predicted_labels.cpu().numpy())

                result["id"] = ids
                result["input_sentence"] = actual_words
                result["label"] = emo_label.cpu().numpy()
                result["prediction"] = predicted_labels.cpu().numpy()

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        columns = ["id", "input_sentence", "label", "prediction"]
        if self.config.use_kt:
            file_name = "/results_{}_kt-{}({})-dropout({})-batchsize({}).csv".format(\
                self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size)
        else:
            file_name = "/resutls_{}_dropout({})-batchsize({}).csv".format(self.config.model, self.config.dropout, self.config.batch_size)
        
        with open(os.getcwd() + file_name, "w") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        
        # Total results log
        accuracy = get_accuracy(y_true, y_pred)
        eval_values = get_metrics(y_true, y_pred)

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

        text_weight, video_weight, audio_weight = [], [], []

        y_true, y_pred, y_pred_dynamic = [], [], []
        tcp_pred = []
        eval_loss, eval_conf_loss, dynamic_eval_loss = [], [], []

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
                        text_weight=None, video_weight=None, audio_weight=None, training=False)
                
                # Text Masking Fusion Model
                _, predicted_scores_t, predicted_labels_t, hidden_state_t = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality="text",\
                        text_weight=None, video_weight=None, audio_weight=None, training=False)

                # Video Masking Fusion Model
                _, predicted_scores_v, predicted_labels_v, hidden_state_v = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality="video",\
                        text_weight=None, video_weight=None, audio_weight=None, training=False)
                
                # Audio Masking Fusion Model
                _, predicted_scores_a, predicted_labels_a, hidden_state_a = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask,labels=emo_label, masked_modality="audio",\
                        text_weight=None, video_weight=None, audio_weight=None, training=False)
                
                
                emo_label = emo_label.type(torch.float)

                # Confidence Model
                predicted_tcp, _ = self.confidence_model(hidden_state)
                predicted_tcp_t, _ = self.confidence_model(hidden_state_t)
                predicted_tcp_v, _ = self.confidence_model(hidden_state_v)
                predicted_tcp_a, _ = self.confidence_model(hidden_state_a)

                predicted_tcp, predicted_tcp_t, predicted_tcp_v, predicted_tcp_a = \
                    predicted_tcp.squeeze(), predicted_tcp_t.squeeze(), predicted_tcp_v.squeeze(), predicted_tcp_a.squeeze()
                
                
                # Calculate weight - method 2
                text_weight.append(torch.sub(predicted_tcp.item(), predicted_tcp_t.item()))
                video_weight.append(torch.sub(predicted_tcp.item(), predicted_tcp_v.item()))
                audio_weight.append(torch.sub(predicted_tcp.item(), predicted_tcp_a.item()))

                # Calculate loss
                conf_loss = self.get_conf_loss(predicted_scores, emo_label, predicted_tcp)

                eval_loss.append(loss.item())
                eval_conf_loss.append(conf_loss.item())

                y_pred.append(predicted_labels.detach().cpu().numpy())
                y_true.append(emo_label.detach().cpu().numpy())
                tcp_pred.append(predicted_tcp.detach().cpu().numpy())

            eval_loss = np.mean(eval_loss)
            eval_conf_loss = np.mean(eval_conf_loss)
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
            
            
            # TODO: adaptive confidence 사용 여부 이용해서 조건문 달기.
            # Make modality-independenct weight score
            scaler = MinMaxScaler()
            text_weight = scaler.fit_transform(np.array(text_weight).reshape(-1, 1)).squeeze()
            video_weight = scaler.fit_transform(np.array(video_weight).reshape(-1, 1)).squeeze()
            audio_weight = scaler.fit_transform(np.array(audio_weight).reshape(-1, 1)).squeeze()
            
            results = list()

            for idx, batch in enumerate(tqdm(self.dataloader)):
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

                emo_label = emo_label.type(torch.float)


                # confidence-aware dynamic weighted fusion model
                dynamic_loss, dynamic_preds, dynamic_labels, _ = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, \
                        masked_modality=None, text_weight=text_weight[idx], video_weight=video_weight[idx], audio_weight=audio_weight[idx], training=False)


                dynamic_eval_loss.append(dynamic_loss.item())
                
                y_pred_dynamic.append(dynamic_labels.detach().cpu().numpy())

                result["id"] = ids[0]
                result["input_sentence"] = actual_words[0]
                result["label"] = emo_label.detach().cpu().numpy()[0]
                result["prediction"] = y_pred[idx]
                result["confidence"] = tcp_pred[idx]
                result["confidence-t"] = text_weight[idx]
                result["confidence-v"] = video_weight[idx]
                result["confidence-a"] = audio_weight[idx]
                result["prediction-dynamic"] = dynamic_labels.detach().cpu().numpy()[0]
                results.append(result)


        columns = ["id", "input_sentence", "label", "prediction", "confidence", "confidence-t", "confidence-v", "confidence-a", "prediction-dynamic"]
        if self.config.use_kt:
            file_name = "/results_kt-{}({})-dropout({})-confidNet-dropout({})-batchsize({}).csv".format(\
            self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.conf_dropout, self.config.batch_size)
        else:
            file_name = "/results_{}-dropout({})-confidNet-dropout({})-batchsize({}).csv".format(\
                self.config.model, self.config.dropout, self.config.conf_dropout, self.config.batch_size)

        with open(os.getcwd() + file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        

        
        dynamic_eval_loss = np.mean(dynamic_eval_loss)
        y_pred_dynamic = np.concatenate(y_pred_dynamic, axis=0).squeeze()

        accuracy = get_accuracy(y_true, y_pred)
        dynamic_accuracy = get_accuracy(y_true, y_pred_dynamic)
        eval_values = get_metrics(y_true, y_pred)
        dynamic_eval_values = get_metrics(y_true, y_pred_dynamic)

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
            "recall": eval_values['recall'],
            "f1": eval_values['f1'],
            "dynamic_model_loss": dynamic_eval_loss,
            "dynamic_model_accuracy": dynamic_accuracy,
            "dynamic_model_precision": dynamic_eval_values['precision'],
            "dynamic_model_recall": dynamic_eval_values['recall'],
            "dynamic_model_f1": dynamic_eval_values['f1']
        }
        
        if self.config.use_kt:
            json_name = "/results_kt-{}({})-dropout({})-confidNet-dropout({})-batchsize({}).json".format(\
            self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.conf_dropout, self.config.batch_size)
        else:
            json_name = "/results_{}-dropout({})-confidNet-dropout({})-batchsize({}).json".format(\
                self.config.model, self.config.dropout, self.config.conf_dropout, self.config.batch_size)

        with open(os.getcwd() + json_name, 'w') as f:
            json.dump(total_results, f)



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
    test_config.batch_size = 1
    test_data_loader = get_loader(test_config, shuffle=False)

    tester = Inference(test_config, test_data_loader, checkpoint=args.checkpoint)

    if test_config.use_confidNet:
        tester.inference_with_confidnet()
    else:
        tester.inference()


if __name__ == "__main__":
    main()