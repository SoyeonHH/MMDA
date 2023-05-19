import numpy as np
import random
from numpy import exp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from sklearn.preprocessing import MinMaxScaler

from utils import to_gpu, to_cpu, DiffLoss, MSE, SIMSE, CMD
from utils.functions import *
from utils import ReverseLayerF, getBinaryTensor

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

class ConfidenceRegressionNetwork(nn.Module):
    def __init__(self, config, input_dims, num_classes=1, dropout=0.1):
        super(ConfidenceRegressionNetwork, self).__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes))

        # Transform features by scaling them to [0, 1]
        self.scaler = MinMaxScaler()
    
    def forward(self, seq_input):
        out = self.mlp(seq_input)

        self.scaler.fit(out.cpu().detach().numpy())
        scaled_out = self.scaler.transform(out.cpu().detach().numpy())
        scaled_out = torch.from_numpy(scaled_out).to(self.config.device).squeeze()

        return out, scaled_out

class MISA(nn.Module):
    """MISA model for CMU-MOSEI emotion multi-label classification"""
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()

        ## Initialize the model

        if self.config.extractor == 'transformer':
            # TODO: Implement transformer encoder for feature extractors
            print("To do implement tranformer encoder")
            exit()

        else:
            rnn = nn.LSTM if self.config.rnncell == 'lstm' else nn.GRU

            if self.config.use_bert:

                # Initializing a BERT bert-base-uncased style configuration
                bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
                self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
            else:
                self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
                self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
                self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
            
            self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
            self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
            
            self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
            self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)


        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


        ##########################################
        # unimodal classifiers
        ##########################################
        # if self.config.freeze == 'False':
        #     self.classifier_t = EmotionClassifier(config.hidden_size, num_classes=output_size,dropout=0.1)
        #     self.classifier_v = EmotionClassifier(config.hidden_size, num_classes=output_size,dropout=0.1)
        #     self.classifier_a = EmotionClassifier(config.hidden_size, num_classes=output_size,dropout=0.1)


        # self.confid_t = ConfidenceRegressionNetwork(config.hidden_size)
        # self.confid_v = ConfidenceRegressionNetwork(config.hidden_size)
        # self.confid_a = ConfidenceRegressionNetwork(config.hidden_size)



        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))

        self.confidence = ConfidenceRegressionNetwork(config, config.hidden_size*6)
        self.classifier = EmotionClassifier(config.hidden_size*6, num_classes=output_size,dropout=dropout_rate)
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

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
        """
        visual: (L, B, Dv)
        aucoustic: (L, B, Da)
        bert_sent: (B, L)

        utterence_text: (B, Dt)
        utterence_video: (B, 2 * Dv)
        utterence_audio: (B, 2 * Da)
        """
        
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


        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)


        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()

        # Modalilty masking before fusion with zero padding
        if masked_modality is not None:
            if "text" in masked_modality:
                self.utt_private_t = torch.zeros_like(self.utt_private_t)
                self.utt_shared_t = torch.zeros_like(self.utt_shared_t)
            if "video" in masked_modality:
                self.utt_private_v = torch.zeros_like(self.utt_private_v)
                self.utt_shared_v = torch.zeros_like(self.utt_shared_v)
            if "audio" in masked_modality:
                self.utt_private_a = torch.zeros_like(self.utt_private_a)
                self.utt_shared_a = torch.zeros_like(self.utt_shared_a)
        
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        h_mask = torch.ones_like(h)

        # decoder_output = self.decoder(label_input, h, label_mask, h_mask)
        predicted_scores = self.classifier(h)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores, self.config.threshold)

        # loss
        cls_loss = get_cls_loss(self.config, predicted_scores, labels)
        kt_loss = get_kt_loss(utterance_text, utterance_video, utterance_audio, labels, dynamic_weight=dynamic_weights)
        domain_loss = get_domain_loss(self.config, self.domain_label_t, self.domain_label_v, self.domain_label_a)
        cmd_loss = get_cmd_loss(self.config, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a)
        diff_loss = get_diff_loss([self.utt_shared_t, self.utt_shared_v, self.utt_shared_a], [self.utt_private_t, self.utt_private_v, self.utt_private_a])
        recon_loss = get_recon_loss([self.utt_t_recon, self.utt_v_recon, self.utt_a_recon], [self.utt_t_orig, self.utt_v_orig, self.utt_a_orig])

        if self.config.use_cmd_sim:
            similarity_loss = cmd_loss
        else:
            similarity_loss = domain_loss

        if training:
            loss = cls_loss + \
                self.config.diff_weight * diff_loss + \
                self.config.sim_weight * similarity_loss + \
                self.config.recon_weight * recon_loss

            if self.config.use_kt:
                loss += self.config.kt_weight * kt_loss
        else:
            loss = cls_loss

        return loss, predicted_scores, predicted_labels, h


    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)        
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


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


class TFN(nn.Module):
    
    def __init__(self, config):
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
        super(TFN, self).__init__()

        # Configuration
        self.config = config
        
        self.text_in = config.embedding_size
        self.video_in = config.visual_size
        self.audio_in = config.acoustic_size

        self.text_hidden = 128
        self.video_hidden = 32
        self.audio_hidden = 32

        self.text_out = 64
        self.post_fusion_dim = config.hidden_size

        self.text_dropout = 0.3
        self.video_dropout = 0.3
        self.audio_dropout = 0.3
        self.post_fusion_dropout = 0.3

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_hidden, dropout=self.audio_dropout)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_hidden, dropout=self.video_dropout)
        self.embed = nn.Embedding(len(config.word2id), self.text_in)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_dropout)

        # define the post-fusion layers
        self.post_fusion_dropout = nn.Dropout(self.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        
        # define the classifier
        self.classifier = EmotionClassifier(self.post_fusion_dim, config.num_classes)
        self.ml_loss = nn.BCELoss()


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

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.config.device), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.config.device), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.config.device), requires_grad=False), text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)
        
        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_2 = F.relu(self.post_fusion_layer_2(post_fusion_1))

        # apply the classifier
        predicted_scores = self.classifier(post_fusion_2)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores, self.config.threshold)
        labels = labels.view(-1, self.config.num_classes)

        # predicted_score = predicted_scores.flatten()
        # labels = labels.flatten()

        # loss
        # cls_loss = get_cls_loss(predicted_scores, labels)
        cls_loss = self.ml_loss(predicted_scores, labels)

        if training and self.config.use_kt:
            kt_loss = get_kt_loss(text_h, video_h, audio_h, labels, dynamic_weight=dynamic_weights)
            loss = cls_loss + self.config.kt_weight * kt_loss
        else:
            loss = cls_loss
        
        return loss, predicted_scores, predicted_labels, post_fusion_2