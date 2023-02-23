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
            self.model.load_state_dict(load_model(config, name=config.model))

        if self.confidence_model is None:
            self.confidence_model = getattr(models, "ConfidenceRegressionNetwork")(self.config, self.config.hidden_size*6, \
                num_classes=1, dropout=self.config.conf_dropout)
            self.confidence_model.load_state_dict(load_model(config, name="confidNet"))
        
        self.model = self.model.to(self.device)
        self.confidence_model = self.confidence_model.to(self.device)
    
    def inference(self):
        print("Start inference...")
        self.criterion = criterion = nn.BCELoss(reduction="mean")
        self.loss_mcp = nn.CrossEntropyLoss(reduction="mean")
        self.loss_tcp = nn.MSELoss(reduction="mean")

        self.model.eval()
        self.confidence_model.eval()

        y_true, y_pred, y_pred_dynamic = [], [], []
        eval_loss, eval_conf_loss, dynamic_eval_loss = [], [], []

        results = list()

        with torch.no_grad():

            for batch in tqdm(self.dataloader):
                result = dict()
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
                predicted_scores, predicted_labels, hidden_state = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, label_input, label_mask, masked_modality=None)
                
                # Text Masking Fusion Model
                predicted_scores_t, predicted_labels_t, hidden_state_t = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, label_input, label_mask, masked_modality="text")

                # Video Masking Fusion Model
                predicted_scores_v, predicted_labels_v, hidden_state_v = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, label_input, label_mask, masked_modality="video")
                
                # Audio Masking Fusion Model
                predicted_scores_a, predicted_labels_a, hidden_state_a = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask,label_input, label_mask, masked_modality="audio")
                
                
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

                # TODO: Add confidence-aware dynamic weighted fusion model
                dynamic_preds, dynamic_labels, _ = \
                    self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, label_input, label_mask, \
                        masked_modality=None, text_weight=predicted_tcp_t, video_weight=predicted_tcp_v, audio_weight=predicted_tcp_a)

                # TODO: Add confidence-aware dynamic Cross-modal Knowledge-based fusion model


                cls_loss = self.get_cls_loss(predicted_scores, emo_label)
                dynamic_cls_loss = self.get_cls_loss(dynamic_preds, emo_label)

                conf_loss = self.get_conf_loss(predicted_scores, emo_label, predicted_tcp)

                eval_loss.append(cls_loss.item())
                dynamic_eval_loss.append(dynamic_cls_loss.item())
                eval_conf_loss.aapend(conf_loss.item())

                y_pred.append(predicted_labels.detach().cpu().numpy())
                y_pred_dynamic.append(dynamic_labels.detach().cpu().numpy())
                y_true.append(emo_label.detach().cpu().numpy())

                result["id"] = ids[0]
                result["input_sentence"] = actual_words[0]
                result["label"] = emo_label.detach().cpu().numpy()[0]
                result["prediction"] = predicted_labels.detach().cpu().numpy()[0]
                result["confidence"] = predicted_tcp.detach().cpu().numpy()
                result["confidence-t"] = predicted_tcp_t.detach().cpu().numpy()
                result["confidence-v"] = predicted_tcp_v.detach().cpu().numpy()
                result["confidence-a"] = predicted_tcp_a.detach().cpu().numpy()
                results.append(result)

                eval_values = get_metrics(emo_label.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
                eval_values_dynamic = get_metrics(emo_label.detach().cpu().numpy(), dynamic_labels.detach().cpu().numpy())

                # wandb.log(
                #     (
                #         {
                #             "test_loss": loss.item(),
                #             "test_conf_loss": conf_loss.item(),
                #             "test_accuracy": eval_values['acc'],
                #             "test_precision": eval_values['precision'],
                #             "test_recall": eval_values['recall'],
                #             "test_f1": eval_values['f1'],
                #             "confidence score": predicted_tcp.detach().cpu().numpy(),
                #             "confidence score-t": predicted_tcp_t.detach().cpu().numpy(),
                #             "confidence score-v": predicted_tcp_v.detach().cpu().numpy(),
                #             "confidence score-a": predicted_tcp_a.detach().cpu().numpy()  
                #         }
                #     )
                # )

        columns = ["id", "input_sentence", "label", "prediction", "confidence", "confidence-t", "confidence-v", "confidence-a"]
        if self.config.conf_scale:
            file_name = "/MMDA/results_scaled.csv"
        else:
            file_name = "/MMDA/results.csv"

        with open(os.getcwd() + file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        

        eval_loss = np.mean(eval_loss)
        eval_conf_loss = np.mean(eval_conf_loss)
        dynamic_eval_loss = np.mean(dynamic_eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        y_pred_dynamic = np.concatenate(y_pred_dynamic, axis=0).squeeze()

        accuracy = get_accuracy(y_true, y_pred)
        dynamic_accuracy = get_accuracy(y_true, y_pred_dynamic)

        print("="*50)
        print("Loss: {:.4f}, Conf Loss: {:.4f}, Accuracy: {:.4f}".format(eval_loss, eval_conf_loss, accuracy))
        eval_values = get_metrics(y_true, y_pred)
        dynamic_eval_values = get_metrics(y_true, y_pred_dynamic)
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
        with open(os.getcwd() + "/MMDA/results.json", 'w') as f:
            json.dump(total_results, f)



    def get_cls_loss(self, predicted_scores, emo_label):
        if self.config.data == "ur_funny":
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

        mcp_loss = self.loss_mcp(pred, truth) * self.config.mcp_weight

        if self.config.use_mcp:
            return torch.add(tcp_loss, mcp_loss, alpha=self.train_config.mcp_weight)
        else:
            return tcp_loss


def main():

    # Setting testing log
    args = config.get_config()
    wandb.init(project="MMDA-Test")
    wandb.config.update(args)
    args.data = 'mosei'

    test_config = config.get_config(mode='test')
    test_config.batch_size = 1
    test_data_loader = get_loader(test_config, shuffle=False)

    tester = Inference(test_config, test_data_loader)
    tester.inference()


if __name__ == "__main__":
    main()