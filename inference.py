'''
Source: input dataset, dataloader, model, config, checkpoint
Output: prediction, confidence, modality-masking confidence scores
functions:
1. load dataset - id, input_sentence, label
2. load model - model, config, checkpoint
3. load dataloader - dataloader
4. return id, input_sentence, label, prediction, confidence, modality-masking confidence
5. save them as csv file
'''

import os
import sys
import math
from math import isnan
import re
import pickle
import json
from collections import defaultdict

from create_dataset import PAD
from solver import *
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from data_loader import get_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import config as config
from utils.tools import *
from utils.eval import *
from MISA.utils.functions import *
import csv
import warnings

warnings.filterwarnings("ignore")

os.chdir(os.getcwd())
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from MISA.utils import to_gpu, to_cpu, time_desc_decorator
from MISA.models import MISA
from TAILOR.models import TAILOR
from EarlyFusion.models import EarlyFusion
from TFN.models import TFN

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Inference(object):
    def __init__(self, config, dataloader, model=None, confidence_model=None, checkpoint=None, dkt=False):
        self.config = config
        self.dataloader = dataloader
        self.checkpoint = checkpoint
        self.dkt = dkt
        self.device = torch.device(config.device)

        if model is None:
            if config.use_kt:
                self.model = load_model(config, dynamicKT=True)
            else:
                self.model = load_model(config)
        else:
            self.model = model
            
        self.model = self.model.to(self.device)
        

        if confidence_model is not None:
            self.confidence_model = confidence_model
            self.confidence_model = self.confidence_model.to(self.device)
        
        
    
    def inference(self):
        print("Start inference...")
        self.model.eval()

        y_true, y_pred = [], []
        pred_tv, pred_ta, pred_va, pred_t, pred_v, pred_a = [], [], [], [], [], []
        eval_loss = []
        results = defaultdict(list)

        if not os.path.exists('results'):
            os.mkdir('results')
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                self.model.zero_grad()
                # result = dict()

                actual_words, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids,\
                    visual_mask, audio_mask, text_mask, labels_embedding, label_mask = batch

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
                visual_mask, audio_mask, text_mask, labels_embedding, label_mask = \
                    to_gpu(visual_mask), to_gpu(audio_mask), to_gpu(text_mask), to_gpu(labels_embedding), to_gpu(label_mask)

                if self.config.model == "TAILOR":
                    loss, predicted_scores, predicted_labels, hidden_state = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, dynamic_weight=None, training=False)
                else:
                    loss, predicted_scores, predicted_labels, hidden_state = self.model(t, v, a, l, \
                        bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, \
                            dynamic_weights=None, training=False)
                
                eval_loss.append(loss.item())
                y_true.append(emo_label.cpu().numpy())
                y_pred.append(predicted_labels.cpu().numpy())

                pred_list, tcp_list = self.get_masking_results(t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, \
                        visual_mask, audio_mask, text_mask, labels_embedding, label_mask)

                pred_va.append(pred_list[0].cpu().numpy())
                pred_ta.append(pred_list[1].cpu().numpy())
                pred_tv.append(pred_list[2].cpu().numpy())
                pred_t.append(pred_list[3].cpu().numpy())
                pred_v.append(pred_list[4].cpu().numpy())
                pred_a.append(pred_list[5].cpu().numpy())

                results["id"].extend(ids)
                results["input_sentence"].extend(actual_words)
                results["label"].extend(emo_label.detach().cpu().numpy())
                results["prediction"].extend(predicted_labels.detach().cpu().numpy())
                results["predicted_scores"].extend(predicted_scores.detach().cpu().numpy())
                
                results["pred_AV"].extend(pred_list[0].cpu().numpy())
                results["pred_TA"].extend(pred_list[1].cpu().numpy())
                results["pred_TV"].extend(pred_list[2].cpu().numpy())
                results["pred_T"].extend(pred_list[3].cpu().numpy())
                results["pred_V"].extend(pred_list[4].cpu().numpy())
                results["pred_A"].extend(pred_list[5].cpu().numpy())
            
                results["tcp_TVA"].extend(get_tcp_target(emo_label, predicted_scores).detach().cpu().numpy())
                results["tcp_AV"].extend(tcp_list[0].detach().cpu().numpy())
                results["tcp_TA"].extend(tcp_list[1].detach().cpu().numpy())
                results["tcp_TV"].extend(tcp_list[2].detach().cpu().numpy())
                results["tcp_T"].extend(tcp_list[3].detach().cpu().numpy())
                results["tcp_V"].extend(tcp_list[4].detach().cpu().numpy())
                results["tcp_A"].extend(tcp_list[5].detach().cpu().numpy())

                # result["id"] = ids[0]
                # result["input_sentence"] = actual_words[0]
                # result["label"] = emo_label.cpu().numpy()[0]
                # result["prediction"] = predicted_labels.cpu().numpy()[0]
                # result["Original_Loss"] = hidden_state[0][0].item()
                # result["T_Masked_Loss"] = hidden_state[0][1].item()
                # result["V_Masked_Loss"] = hidden_state[0][3].item()
                # result["A_Masked_Loss"] = hidden_state[0][2].item()
                
        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        if self.config.use_kt:
            if self.config.kt_model == "Dynamic-tcp" and self.config.warm_start == True:
                csv_file_name = os.getcwd() + "/results/results_{}_{}_{}({})_dropout({})_batchsize({})_epoch({})_warmstart.csv".format(\
                    self.config.data, self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size, self.config.n_epoch)
            else:
                csv_file_name = os.getcwd() + "/results/results_{}_{}_{}({})_dropout({})_batchsize({})_epoch({}).csv".format(\
                    self.config.data, self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size, self.config.n_epoch)
        else:
            csv_file_name = os.getcwd() + "/results/results_{}_{}_dropout({})_batchsize({})_epoch({}).csv".format(\
                self.config.data, self.config.model, self.config.dropout, self.config.batch_size, self.config.n_epoch)

        # columns = ["id", "input_sentence", "label", "prediction", "Original_Loss", "T_Masked_Loss", "V_Masked_Loss","A_Masked_Loss"]
        # if self.config.use_kt:
        #     file_name = "/results_{}_kt-{}({})-dropout({})-batchsize({}).csv".format(\
        #         self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size)
        # else:
        #     file_name = "/results_{}_dropout({})-batchsize({}).csv".format(self.config.model, self.config.dropout, self.config.batch_size)
        
        with open(csv_file_name, 'w') as f:
            key_list = list(results.keys())
            writer = csv.writer(f)
            writer.writerow(results.keys())
            for i in range(len(results["id"])):
                writer.writerow([results[x][i] for x in key_list])
        
        # Total results log
        accuracy = get_accuracy(y_true, y_pred)
        eval_values = get_metrics(y_true, y_pred, average=self.config.eval_mode)

        print("="*50)
        print("Loss: {:.4f}, Accuracy: {:.4f}".format(eval_loss, accuracy))
        print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(eval_values['precision'], eval_values['recall'], eval_values['f1']))
        print("="*50)

        acc_list = [
            get_accuracy(y_true, np.concatenate(pred_va, axis=0).squeeze()),
            get_accuracy(y_true, np.concatenate(pred_ta, axis=0).squeeze()),
            get_accuracy(y_true, np.concatenate(pred_tv, axis=0).squeeze()),
            get_accuracy(y_true, np.concatenate(pred_t, axis=0).squeeze()),
            get_accuracy(y_true, np.concatenate(pred_v, axis=0).squeeze()),
            get_accuracy(y_true, np.concatenate(pred_a, axis=0).squeeze())
        ]

        # Save metric results into json
        total_results = {
            "loss": eval_loss,
            "accuracy": accuracy,
            "precision": eval_values['precision'],
            "recall": eval_values['recall'],
            "f1": eval_values['f1'],
            "acc_va": acc_list[0],
            "acc_ta": acc_list[1],
            "acc_tv": acc_list[2],
            "acc_t": acc_list[3],
            "acc_v": acc_list[4],
            "acc_a": acc_list[5]
        }

        if self.config.use_kt:
            if self.config.kt_model == "Dynamic-tcp" and self.config.warm_start == True:
                json_name = "/results/results_{}_{}_{}({})_dropout({})_batchsize({})_epoch({})_warmstart.json".format(\
                    self.config.data, self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size, self.config.n_epoch)
            else:
                json_name = "/results/results_{}_kt-{}({})-dropout({})-batchsize({}).json".format(\
                    self.config.model, self.config.kt_model, self.config.kt_weight, self.config.dropout, self.config.batch_size)
        else:
            json_name = "/results/results_{}_baseline_dropout({})-batchsize({})_epoch({}).json".format(self.config.model, self.config.dropout, self.config.batch_size, self.config.n_epoch)
        
        with open(os.getcwd() + json_name, "w") as f:
            json.dump(total_results, f, indent=4)

        
    def get_masking_results(self, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, \
                        visual_mask, audio_mask, text_mask, labels_embedding, label_mask):
        
        if self.config.model == "TAILOR":
            _, logit_text_removed, pred_text_removed, z_text_removed = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["text"], training=False)

            _, logit_visual_removed, pred_visual_removed, z_visual_removed = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["video"], training=False)
            
            _, logit_audio_removed, pred_audio_removed, z_audio_removed = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["audio"], training=False)
            
            _, logit_only_text, pred_only_text, z_only_text = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["video", "audio"], training=False)
            
            _, logit_only_visual, pred_only_visual, z_only_visual = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["text", "audio"], training=False)
            
            _, logit_only_audio, pred_only_audio, z_only_audio = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["text", "video"], training=False)
        
        else:
            _, logit_text_removed, pred_text_removed, z_text_removed = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["text"], training=False)
            
            _, logit_visual_removed, pred_visual_removed, z_visual_removed = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["video"], training=False)
            
            _, logit_audio_removed, pred_audio_removed, z_audio_removed = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["audio"], training=False)
            
            _, logit_only_text, pred_only_text, z_only_text = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["video", "audio"], training=False)
            
            _, logit_only_visual, pred_only_visual, z_only_visual = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["text", "audio"], training=False)
            
            _, logit_only_audio, pred_only_audio, z_only_audio = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["text", "video"], training=False)

            
        pred_list = [pred_text_removed, pred_visual_removed, pred_audio_removed, pred_only_text, pred_only_visual, pred_only_audio]
        tcp_list = [
            get_tcp_target(emo_label, logit_text_removed),
            get_tcp_target(emo_label, logit_visual_removed),
            get_tcp_target(emo_label, logit_audio_removed),
            get_tcp_target(emo_label, logit_only_text),
            get_tcp_target(emo_label, logit_only_visual),
            get_tcp_target(emo_label, logit_only_audio)
        ]

        return pred_list, tcp_list


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
            return torch.add(tcp_loss, mcp_loss, alpha=self.config.mcp_weight)
        else:
            return tcp_loss


def main():

    # Setting testing log
    args = config.get_config()

    test_config = config.get_config(mode='test')
    # test_config.batch_size = 1
    test_data_loader = get_loader(test_config, shuffle=False)

    tester = Inference(test_config, test_data_loader)

    # if test_config.use_confidNet:
    #     tester.inference_with_confidnet()
    # else:
    tester.inference()


if __name__ == "__main__":
    main()