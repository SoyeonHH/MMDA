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
from utils.tools import *
from utils.eval import *
from MISA.utils.functions import *
import time
import datetime
import wandb

import hypertune
hpt = hypertune.HyperTune()

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from MISA.utils import to_gpu, to_cpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
from MISA.models import MISA
from EarlyFusion.models import EarlyFusion
from TAILOR.models import TAILOR
from TAILOR.optimization import BertAdam
from confidNet import ConfidenceRegressionNetwork

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None, confidnet=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        self.confidnet = confidnet
        
        if torch.cuda.is_available():
            self.device = torch.device(train_config.device)
        else:
            self.device = torch.device("cpu")
        print(f"current device: {self.device}")
    
    # @time_desc_decorator('Build Graph')
    def build(self, cuda=True): 
        
        # Prepare model
        if self.model is None:
            if self.train_config.model == "Early":
                self.model = EarlyFusion(self.train_config, (128, 32, 32), 64, (0.3, 0.3, 0.3, 0.3), 32)
            elif self.train_config.model == "MISA":
                self.model = MISA(self.train_config)
            elif self.train_config.model == "TAILOR":
                self.model = TAILOR.from_pretrained(self.train_config.bert_model, \
                        self.train_config.visual_model, self.train_config.audio_model, self.train_config.cross_model, \
                            self.train_config.decoder_model, task_config=self.train_config)

        # Final list 
        for name, param in self.model.named_parameters():
            param.requires_grad = True

            # Bert freezing customizations 
            if "bertmodel.encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                if layer_num <= (8):
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert:
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False
        
        # Initialize ConfidNet model
        if self.confidnet is not None:
            for para in self.confidnet.parameters():
                para.requires_grad = False
        
        # # Multi-GPU training setting
        # if torch.cuda.device_count() > 1:
        #     self.train_config.n_gpu = torch.cuda.device_count()
        #     print("Let's use", self.train_config.n_gpu, "GPUs!")
        #     self.model = nn.DataParallel(self.model, device_ids=[i for i in range(self.train_config.n_gpu)])
        
        # if torch.cuda.is_available() and cuda:
        self.model.to(self.device)

        if self.is_train:
            self.model.train()
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay)

    def train(self, additional_training=False):
        curr_patience = patience = self.train_config.patience
        num_trials = 1
        
        best_valid_loss = float('inf')
        best_train_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        train_losses = []
        total_start = time.time()

        if additional_training:
            print("Training the fusion model with dynamic weighted kt...")
            n_epoch = self.train_config.n_epoch_dkt
        else:
            n_epoch = self.train_config.n_epoch

        for e in range(n_epoch):
            self.model.train()

            train_loss = []

            for idx, batch in enumerate(tqdm(self.train_data_loader)):
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                _, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids, \
                    visual_mask, audio_mask, text_mask, labels_embedding, label_mask = batch

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
                visual_mask, audio_mask, text_mask, labels_embedding, label_mask = \
                    to_gpu(visual_mask), to_gpu(audio_mask), to_gpu(text_mask), to_gpu(labels_embedding), to_gpu(label_mask)

                # Dynamic weighted kt
                if self.train_config.kt_model == "Dynamic-tcp" and additional_training:
                    dynamic_weight = self.get_dynamic_tcp(t, v, a, y, \
                            emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, \
                                visual_mask, audio_mask, text_mask, labels_embedding, label_mask)
                elif self.train_config.kt_model == "Dynamic-ce":
                    dynamic_weight = self.get_dynamic_ce(t, v, a, y, \
                            emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, \
                                visual_mask, audio_mask, text_mask, labels_embedding, label_mask)
                else:
                    dynamic_weight = None

                # Forward pass
                if self.train_config.model == "TAILOR":
                    loss, y_tilde, predicted_labels, _ = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, dynamic_weight=dynamic_weight, training=True)
                else:
                    loss, y_tilde, predicted_labels, _ = self.model(t, v, a, l, \
                        bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, \
                            dynamic_weights=dynamic_weight, training=True)

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
                save_model(self.train_config, self.model, dynamicKT=True) if additional_training else save_model(self.train_config, self.model)
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
                        "test_f_score": eval_values['f1'],
                        "test_precision": eval_values['precision'],
                        "test_recall": eval_values['recall'],
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
        # Report best model results
        ##########################################

        print("model training is finished.")
        print('='*50)
        print(f'Best epoch: {best_epoch}')
        eval_values_best = get_metrics(best_truths, best_results, average=self.train_config.eval_mode)
        best_acc, best_f1, best_precision, best_recall = \
             eval_values_best['acc'], eval_values_best['f1'], eval_values_best['precision'], eval_values_best['recall']
        print(f'Accuracy: {best_acc}')
        print(f'F1 score: {best_f1}')
        print(f'Precision: {best_precision}')
        print(f'Recall: {best_recall}')
        total_end = time.time()
        total_duration = total_end - total_start
        print(f"Total training time: {total_duration}s, {datetime.timedelta(seconds=total_duration)}")

        return self.model

    
    def eval(self,mode=None, to_print=False):
        assert(mode is not None)

        if hasattr(self.model, 'module'):
            self.model = self.model.module.to(self.device)
        else:
            self.model = self.model.to(self.device)

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

                _, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids, \
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

                if self.train_config.model == "TAILOR":
                    loss, y_tilde, predicted_labels, _ = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, dynamic_weight=None, training=False)
                else:
                    loss, y_tilde, predicted_labels, _ = self.model(t, v, a, l, \
                        bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, \
                            dynamic_weights=None, training=False)

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

    

    def get_dynamic_tcp(self, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, \
                        visual_mask, audio_mask, text_mask, labels_embedding, label_mask):
        
        # Get TCP for each modality
        if self.train_config.model == "TAILOR":
            _, outputs, output_labels, hidden_state = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, training=True)
            target_tcp = get_tcp_target(emo_label, outputs)

            _, tcp = self.confidnet(hidden_state, target_tcp)

            # return the tcp for each maksed modality
            _, _, _, z_text_removed = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["text"], training=True)
            _, tcp_text_removed = self.confidnet(z_text_removed, target_tcp)

            _, _, _, z_video_removed = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["video"], training=True)
            _, tcp_video_removed = self.confidnet(z_video_removed, target_tcp)

            _, _, _, z_audio_removed = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["audio"], training=True)
            _, tcp_audio_removed = self.confidnet(z_audio_removed, target_tcp)
        
        else:
            _, outputs, output_labels, hidden_state = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, training=False)
            target_tcp = get_tcp_target(emo_label, outputs)

            _, tcp = self.confidnet(hidden_state, target_tcp)

            # return the tcp for each maksed modality
            _, _, _, z_text_removed = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["text"], training=False)
            _, tcp_text_removed = self.confidnet(z_text_removed, target_tcp)

            _, _, _, z_video_removed = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["video"], training=False)
            _, tcp_video_removed = self.confidnet(z_video_removed, target_tcp)

            _, _, _, z_audio_removed = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["audio"], training=False)
            _, tcp_audio_removed = self.confidnet(z_audio_removed, target_tcp)

        # Get dynamic weight
        dynamic_weight = []

        if self.train_config.dynamic_method == "threshold":
            dynamic_weight = [[1 if tcp_text_removed[i] > tcp_video_removed[i] else 0 for i in range(len(tcp_text_removed))], \
                                [1 if tcp_text_removed[i] > tcp_audio_removed[i] else 0 for i in range(len(tcp_text_removed))], \
                                [1 if tcp_video_removed[i] > tcp_text_removed[i] else 0 for i in range(len(tcp_text_removed))], \
                                [1 if tcp_video_removed[i] > tcp_audio_removed[i] else 0 for i in range(len(tcp_text_removed))], \
                                [1 if tcp_audio_removed[i] > tcp_text_removed[i] else 0 for i in range(len(tcp_text_removed))], \
                                [1 if tcp_audio_removed[i] > tcp_video_removed[i] else 0 for i in range(len(tcp_text_removed))]]
        
        elif self.train_config.dynamic_method == "ratio":
            dynamic_weight = [[torch.max(torch.zeros_like(tcp_text_removed[i]), tcp_text_removed[i] - tcp_video_removed[i]) for i in range(len(tcp_text_removed))], \
                                [torch.max(torch.zeros_like(tcp_text_removed[i]), tcp_text_removed[i] - tcp_audio_removed[i]) for i in range(len(tcp_text_removed))], \
                                [torch.max(torch.zeros_like(tcp_text_removed[i]), tcp_video_removed[i] - tcp_text_removed[i]) for i in range(len(tcp_text_removed))], \
                                [torch.max(torch.zeros_like(tcp_text_removed[i]), tcp_video_removed[i] - tcp_audio_removed[i]) for i in range(len(tcp_text_removed))], \
                                [torch.max(torch.zeros_like(tcp_text_removed[i]), tcp_audio_removed[i] - tcp_text_removed[i]) for i in range(len(tcp_text_removed))], \
                                [torch.max(torch.zeros_like(tcp_text_removed[i]), tcp_audio_removed[i] - tcp_video_removed[i]) for i in range(len(tcp_text_removed))]]

            # dynamic_weight = [[tcp_text_removed[i] if tcp_text_removed[i] > tcp_video_removed[i] else 0 for i in range(len(tcp_text_removed))], \
            #                     [tcp_text_removed[i] if tcp_text_removed[i] > tcp_audio_removed[i] else 0 for i in range(len(tcp_text_removed))], \
            #                     [tcp_video_removed[i] if tcp_video_removed[i] > tcp_text_removed[i] else 0 for i in range(len(tcp_text_removed))], \
            #                     [tcp_video_removed[i] if tcp_video_removed[i] > tcp_audio_removed[i] else 0 for i in range(len(tcp_text_removed))], \
            #                     [tcp_audio_removed[i] if tcp_audio_removed[i] > tcp_text_removed[i] else 0 for i in range(len(tcp_text_removed))], \
            #                     [tcp_audio_removed[i] if tcp_audio_removed[i] > tcp_video_removed[i] else 0 for i in range(len(tcp_text_removed))]]
            
        elif self.train_config.dynamic_method == "noise_level":
            dynamic_weight = [[tcp_text_removed[i] if tcp_text_removed[i] > tcp[i] else 0 for i in range(len(tcp_text_removed))], \
                                [tcp_text_removed[i] if tcp_text_removed[i] > tcp[i] else 0 for i in range(len(tcp_text_removed))], \
                                [tcp_video_removed[i] if tcp_video_removed[i] > tcp[i] else 0 for i in range(len(tcp_text_removed))], \
                                [tcp_video_removed[i] if tcp_video_removed[i] > tcp[i] else 0 for i in range(len(tcp_text_removed))], \
                                [tcp_audio_removed[i] if tcp_audio_removed[i] > tcp[i] else 0 for i in range(len(tcp_text_removed))], \
                                [tcp_audio_removed[i] if tcp_audio_removed[i] > tcp[i] else 0 for i in range(len(tcp_text_removed))]]
            
            
        dynamic_weight = torch.tensor(dynamic_weight, dtype=torch.float).to(self.device)

        return dynamic_weight

    def get_dynamic_ce(self, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, \
                       visual_mask, audio_mask, text_mask, labels_embedding, label_mask):

        if self.train_config.model == "TAILOR":
            _, prob_all, _, _ = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=None, training=True)

            _, prob_text_removed, _, _ = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["text"], training=True)
            
            _, prob_video_removed, _, _ = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["video"], training=True)
            
            _, prob_audio_removed, _, _ = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, masked_modality=["audio"], training=True)
        
        else:
            _, prob_all, _, _ = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, training=False)
            
            _, prob_text_removed, _, _ = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["text"], training=False)

            _, prob_video_removed, _, _ = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["video"], training=False)

            _, prob_audio_removed, _, _ = self.model(t, v, a, l, \
                bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=["audio"], training=False)
            
        
        t_mask_loss = binary_ce(prob_all, prob_text_removed)
        v_mask_loss = binary_ce(prob_all, prob_video_removed)
        a_mask_loss = binary_ce(prob_all, prob_audio_removed)

        if self.train_config.dynamic_method == "threshold":
            dynamic_weight = [[0 if t_mask_loss[i] > v_mask_loss[i] else 1 for i in range(len(t_mask_loss))], \
                [0 if t_mask_loss[i] > a_mask_loss[i] else 1 for i in range(len(t_mask_loss))], \
                [0 if v_mask_loss[i] > t_mask_loss[i] else 1 for i in range(len(t_mask_loss))], \
                [0 if v_mask_loss[i] > a_mask_loss[i] else 1 for i in range(len(t_mask_loss))], \
                [0 if a_mask_loss[i] > t_mask_loss[i] else 1 for i in range(len(t_mask_loss))], \
                [0 if a_mask_loss[i] > v_mask_loss[i] else 1 for i in range(len(t_mask_loss))]]
        
        elif self.train_config.dynamic_method == "ratio":
            # dynamic_weight = [[t_mask_loss[i] / v_mask_loss[i] if t_mask_loss[i] > v_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
            #     [t_mask_loss[i] / a_mask_loss[i] if t_mask_loss[i] > a_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
            #     [v_mask_loss[i] / t_mask_loss[i] if v_mask_loss[i] > t_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
            #     [v_mask_loss[i] / a_mask_loss[i] if v_mask_loss[i] > a_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
            #     [a_mask_loss[i] / t_mask_loss[i] if a_mask_loss[i] > t_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
            #     [a_mask_loss[i] / v_mask_loss[i] if a_mask_loss[i] > v_mask_loss[i] else 0 for i in range(len(t_mask_loss))]]
            
            # TODO: Train again with this ratio
            dynamic_weight = [[t_mask_loss[i] / v_mask_loss[i] if t_mask_loss[i] > v_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
                [t_mask_loss[i] if t_mask_loss[i] > a_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
                [v_mask_loss[i] if v_mask_loss[i] > t_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
                [v_mask_loss[i] if v_mask_loss[i] > a_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
                [a_mask_loss[i] if a_mask_loss[i] > t_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
                [a_mask_loss[i] if a_mask_loss[i] > v_mask_loss[i] else 0 for i in range(len(t_mask_loss))]]

        dynamic_weight = torch.tensor(dynamic_weight, dtype=torch.float).to(self.device)

        return dynamic_weight
    


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
            self.model = load_model(self.config)

        self.model.eval()

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize confidence network
        if self.confidnet is None:
            self.confidnet = ConfidenceRegressionNetwork(self.config, input_dims=self.config.hidden_size, num_classes=1, dropout=self.config.conf_dropout)
            self.confidnet = self.confidnet.to(self.device)

        self.optimizer = torch.optim.Adam(self.confidnet.parameters(), lr=self.config.conf_lr)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.config.use_bert:
            if self.config.pretrained_emb is not None:
                self.model.embed.weight.data = self.config.pretrained_emb
            self.model.embed.requires_grad = False
        
        self.model = self.model.to(self.device)
        

    def train(self):
        print("Training Confidence Network...")
        self.confidnet.train()
        train_results = []
        best_valid_loss = float('inf')

        for epoch in range(self.config.n_epoch_conf):
            train_losses = []

            for i, batch in enumerate(tqdm(self.train_data_loader)):
                self.optimizer.zero_grad()

                actual_words, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids, \
                    visual_mask, audio_mask, text_mask, labels_embedding, label_mask = batch

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
                visual_mask, audio_mask, text_mask, labels_embedding, label_mask = \
                    to_gpu(visual_mask), to_gpu(audio_mask), to_gpu(text_mask), to_gpu(labels_embedding), to_gpu(label_mask)

                # Get the output from the classification model
                if self.config.model == "TAILOR":
                    _, outputs, output_labels, hidden_state = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, dynamic_weight=None, training=False)
                else:
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
            valid_loss = self.eval(mode="dev")

            print("-" * 100)
            print("Epochs: {}, Valid loss: {}".format(epoch, valid_loss))
            print("-" * 100)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                print("Saving the best model...")
                save_model(self.config, self.confidnet, confidNet=True)

            # wandb.log({"train_loss": train_avg_loss, "valid_loss": valid_loss})


        # Model Test
        print("Testing the model...")
        test_loss = self.eval(mode="test")
        print('='*50)
        print(f'Best epoch: {best_epoch}')
        print(f'Best valid loss: {best_valid_loss}')
        print(f'Test loss: {test_loss}')
        print('='*50)

        return self.confidnet

    
    def eval(self, mode=None):
        self.confidnet.eval()
        eval_losses = []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader
        
        with torch.no_grad():

            for batch in dataloader:
                # self.model.zero_grad()
                # self.confidnet.zero_grad()

                actual_words, t, v, a, y, emo_label, l, bert_sent, bert_sent_type, bert_sent_mask, ids, \
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
                    _, outputs, output_labels, hidden_state = self.model(t, text_mask, v, visual_mask, a, audio_mask, \
                            labels_embedding, label_mask, groundTruth_labels=emo_label, dynamic_weight=None, training=True)
                else:
                    _, outputs, output_labels, hidden_state = self.model(t, v, a, l, \
                        bert_sent, bert_sent_type, bert_sent_mask, labels=emo_label, masked_modality=None, training=False)
                    
                target_tcp = get_tcp_target(emo_label, outputs)
                loss, predicts = self.confidnet(hidden_state, target_tcp)

                eval_losses.append(loss.item())

        eval_avg_loss = np.mean(eval_losses)

        return eval_avg_loss