#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
__file__    :   TextRCNN.PY
__time__    :   2022/05/30 16:29:34
__author__  :   yangning
__copyright__   :  Copyright 2022
'''

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

class TextRCNNPredict:
    def __init__(self):

        self.config = Config()
        self.model = Model(self.config)        
        self.use_word = False
        self._common_config()
    
    def _common_config(self):
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  

    def _sentence_tokenizer(self,sentence1,sentence2):
        if self.use_word:
            tokenizer = lambda x: x.split(' ')
        else:
            tokenizer = lambda x: [y for y in x]

        vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        words_line = []
        sentence = sentence1 + "&&" + sentence2

        token = tokenizer(sentence)
        seq_len = len(token)
        if self.config.pad_size:
            if len(token) < self.config.pad_size:
                token.extend([PAD] * (self.config.pad_size - len(token)))
            else:
                token = token[:self.config.pad_size]
                seq_len = self.config.pad_size

        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
            
        return [(words_line,1,seq_len)]                            

    def predict(self,sentence1,sentence2):
        
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.to(self.config.device)
        self.model.eval()

        sentence = self._sentence_tokenizer(sentence1,sentence2)
        data_iterator = DatasetIterater(sentence,1,self.config.device)

        with torch.no_grad():
            for texts, labels in data_iterator:
                outputs = self.model(texts)
                prob = outputs.softmax(-1).cpu().numpy()
                predict = torch.max(outputs.data, 0)[1].cpu().numpy()
                return prob[1]

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches    


class Config(object):

    def __init__(self):
        self.os_path = sys.path[0]
        self.model_name = 'TextRCNN'
        self.model_path = self.os_path + '/Text_relevance/' + self.model_name
        self.vocab_path = self.model_path + '/data/vocab.pkl'
        self.save_path = self.model_path + '/saved_dict/' + self.model_name + '.ckpt'
        self.embedding  = self.model_path + '/data/embedding_SougouNews.npz'

        self.embedding_pretrained = torch.tensor(
            np.load(self.embedding)["embeddings"].astype('float32'))\
            if self.embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 1.0
        self.require_improvement = 1000
        self.num_classes = 2
        self.n_vocab = 0
        self.num_epochs = 10
        self.batch_size = 128
        self.pad_size = 64
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.hidden_size = 256
        self.num_layers = 1

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
