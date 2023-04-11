"""
reference: Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." arXiv preprint arXiv:1707.07250 (2017). https://github.com/Justin1904/TensorFusionNetworks
"""

import numpy as np
from EarlyFusion.functions import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig


class EmotionClassifier(nn.Module):
    def __init__(self, input_dims, num_classes, dropout=0.1):
        super(EmotionClassifier, self).__init__()
        self.dense = nn.Linear(input_dims, num_classes)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_input):
        out = self.dense(seq_input)
        out = self.dropout(out)
        out = self.activation(out)
        return out


class EarlyFusion(nn.Module):
    
    def __init__(self, config, hidden_dims, text_out, dropouts, post_fusion_dim):
        '''
        Args:
            input_dims - a length-3 tuple, contains (text_dim, video_dim, audio_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (text_dropout, video_dropout, audio_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) multi-label classification results
        '''
        super(EarlyFusion, self).__init__()

        # Configuration
        self.config = config
        
        self.text_in = config.embedding_size
        self.video_in = config.visual_size
        self.audio_in = config.acoustic_size

        self.text_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.audio_hidden = hidden_dims[2]

        self.text_out = text_out
        self.post_fusion_dim = post_fusion_dim

        self.text_dropout = dropouts[0]
        self.video_dropout = dropouts[1]
        self.audio_dropout = dropouts[2]
        self.post_fusion_dropout = dropouts[3]

        # define the pre-fusion subnetworks
        rnn = nn.LSTM if self.config.rnncell == 'lstm' else nn.GRU

        if self.config.use_bert:

            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(config.word2id), self.text_in)
            self.trnn1 = rnn(self.text_in, self.text_hidden, bidirectional=True)
            self.trnn2 = rnn(2*self.text_hidden, self.text_out, bidirectional=True)
        
        self.vrnn1 = rnn(self.video_in, self.video_hidden, bidirectional=True)
        self.vrnn2 = rnn(2*self.video_hidden, self.video_hidden, bidirectional=True)
        
        self.arnn1 = rnn(self.audio_in, self.audio_hidden, bidirectional=True)
        self.arnn2 = rnn(2*self.audio_hidden, self.audio_hidden, bidirectional=True)

        # define the post-fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden, self.post_fusion_dim)    # Concatenation
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        
        # define the classifier
        self.classifier = EmotionClassifier(self.post_fusion_dim, config.num_classes)
    
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def forward(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask,
                  labels=None, masked_modality=None, dynamic_weights=None, training=True):
        '''
        visual: (L, B, Dv)
        aucoustic: (L, B, Da)
        bert_sent: (B, L)

        utterence_text: (B, Dt)
        utterence_video: (B, 2 * Dv)
        utterence_audio: (B, 2 * Da)
        '''

        batch_size = lengths.size(0)
        labels = labels.type(torch.float)
        
        if self.config.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent, 
                                         attention_mask=bert_sent_mask, 
                                         token_type_ids=bert_sent_type)      

            bert_output = bert_output[0]

            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

            utterance_text = bert_output
        else:
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)


        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)


        # concatenate the outputs of the subnetworks
        h = torch.cat((utterance_text, utterance_video, utterance_audio), dim=1)

        # apply the post-fusion layers
        h_dropped = self.post_fusion_dropout(h)
        h_1 = F.relu(self.post_fusion_layer_1(h_dropped))
        h_2 = F.relu(self.post_fusion_layer_2(h_1))

        # apply the classifier
        predicted_scores = self.classifier(h_2)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores, self.config.threshold)

        # loss
        cls_loss = get_cls_loss(predicted_scores, labels)
        kt_loss = get_kt_loss(utterance_text, utterance_video, utterance_audio, labels, dynamic_weight=dynamic_weights)

        if training and self.config.use_kt:
            loss = cls_loss + self.config.kt_weight * kt_loss
        else:
            loss = cls_loss
        
        return loss, predicted_scores, predicted_labels, h

        
