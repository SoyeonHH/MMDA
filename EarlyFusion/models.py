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
    
class SubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(SubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


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
        self.post_fusion_dim = config.hidden_size

        self.text_dropout = dropouts[0]
        self.video_dropout = dropouts[1]
        self.audio_dropout = dropouts[2]
        self.post_fusion_dropout = dropouts[3]

        self.ml_loss = nn.BCELoss(reduction="sum")

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_hidden, dropout=self.audio_dropout)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_hidden, dropout=self.video_dropout)
        self.embed = nn.Embedding(len(config.word2id), self.text_in)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_dropout)

       # define the post-fusion layers
        self.post_fusion_dropout = nn.Dropout(self.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        
        # define the classifier
        self.classifier = EmotionClassifier(self.post_fusion_dim, config.num_classes)


    def forward(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask,
                  labels=None, masked_modality=None, dynamic_weights=None, training=True):
        '''
        sentence: (L, B) -> (B, L, Dt) -> (B, Dt)
        visual: (L, B, Dv) -> (B, L, Dv) -> (B, Dv)
        audio: (L, B, Da) -> (B, L, Da) -> (B, Da)
        '''

        batch_size = lengths.size(0)
        labels = labels.type(torch.float)

        # extract features from subnets
        sentences = self.embed(sentences)
        sentences = sentences.view(batch_size, -1, self.text_in)
        visual = visual.view(batch_size, -1, self.video_in)
        acoustic = acoustic.view(batch_size, -1, self.audio_in)

        text_h = self.text_subnet(sentences)
        video_h = self.video_subnet(visual)
        audio_h = self.audio_subnet(acoustic)

        # modality masking
        if masked_modality is not None:
            if "text" in masked_modality:
                text_h = torch.zeros_like(text_h)
            if "video" in masked_modality:
                video_h = torch.zeros_like(video_h)
            if "audio" in masked_modality:
                audio_h = torch.zeros_like(audio_h)

        # concatenate the outputs of the subnetworks
        h = torch.cat((text_h, video_h, audio_h), dim=1)

        # apply the post-fusion layers
        h_dropped = self.post_fusion_dropout(h)
        h_1 = F.relu(self.post_fusion_layer_1(h_dropped))
        h_2 = F.relu(self.post_fusion_layer_2(h_1))

        # apply the classifier
        predicted_scores = self.classifier(h_2)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores, self.config.threshold)

        # predicted_score = predicted_scores.flatten()
        # labels = labels.flatten()

        # loss
        cls_loss = get_cls_loss(predicted_scores, labels)

        if training and self.config.use_kt:
            kt_loss = get_kt_loss(text_h, video_h, audio_h, labels, dynamic_weight=dynamic_weights)
            loss = cls_loss + self.config.kt_weight * kt_loss
        else:
            loss = cls_loss
        
        return loss, predicted_scores, predicted_labels, h_2

        
