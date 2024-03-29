import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import MOSI, MOSEI, UR_FUNNY, PAD, UNK


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MSADataset(Dataset):
    def __init__(self, config):

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: np.array(x[0][0]).shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things


        # labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        bert_details = []
        labels = []
        emo_labels = []
        ids = []
        for sample in batch:
            ids.append(sample[2])
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)
            bert_details.append(encoded_bert_sent)
            # if sample[1] 모든 요소가 0이라면, labels.append(0) else sample[1]에서 null 값은 모두 없앤 후 labels.append(sample[1][0])
            if sample[1].all() == 0.:
                labels.append([sample[1]][0][0])
            else:
                labels.append([np.nan_to_num(sample[1])][0][0])
        if labels[0].size == 7:
            labels = np.array(labels)
            # emo_label: (6) -> [happy, sad, anger, surprise, disgust, fear] 
            #                -> averaged from 3 annotators
            #                -> Should convert to multiple binary class vector (6,1)
            filter_label = labels[:,1:]
            for i in range(filter_label.shape[0]):
                emo_label = np.zeros(6, dtype=np.float32)
                for j, num in enumerate(filter_label[i]):
                    emo_label[j] = 1 if num > 0.0 else 0
                emo_labels.append(emo_label)
            labels = labels[:,0]
        else:
            emo_labels = None


        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        labels = torch.cat([torch.FloatTensor([label]) for label in labels], dim=0)
        emo_labels = torch.from_numpy(np.array(emo_labels))
        # emo_labels = torch.cat([torch.FloatTensor(emo_label) for emo_label in emo_labels], dim=0)

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, visual, acoustic, labels, emo_labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader

class UnAlignedMoseiDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()