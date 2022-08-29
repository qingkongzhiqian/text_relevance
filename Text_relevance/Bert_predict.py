#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
__file__    :   Bert_predict.py
__time__    :   2022/05/31 14:06:16
__author__  :   yangning
__copyright__   :  Copyright 2022
'''

import torch
import sys
import json
from Text_relevance.transformers import BertConfig,BertForSequenceClassification,BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
import copy

class BertPredict:
    def __init__(self):

        self.os_path = sys.path[0]
        self.label_list = ["0", "1"]
        self.task_name = "afqmc"
        self.do_lower_case = True
        self.model_name = 'BERT'
        self.model_name_or_path = self.os_path + '/Text_relevance/' + self.model_name
        self.pred_batch_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_seq_length = 64

        self.config = BertConfig.from_pretrained(self.model_name_or_path,
                                            num_labels=len(self.label_list),
                                            finetuning_task=self.task_name)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path,do_lower_case=self.do_lower_case)        

        self.model = BertForSequenceClassification.from_pretrained(self.model_name_or_path, from_tf=bool('.ckpt' in self.model_name_or_path),config=self.config)
        self.model.to(self.device)
        self.model.eval()

        print ("Bert load finish !")

    def _create_examples(self,sentence1,sentence2):

        guid = "%s-%s" % ("test", 1)
        text_a = sentence1
        text_b = sentence2
        label =  "0"
        return [InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)]

    def _convert_examples_to_features(self,examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
        
        label_map = {label: i for i, label in enumerate(self.label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):

            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(input_ids)
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            label = label_map[example.label]
            features.append(
                InputFeatures(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label=label,
                            input_len=input_len))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
        return dataset                   

    
    def _collate_fn(self,batch):

        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
        max_len = max(all_lens).item()
        all_input_ids = all_input_ids[:, :max_len]
        all_attention_mask = all_attention_mask[:, :max_len]
        all_token_type_ids = all_token_type_ids[:, :max_len]
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels

    def predict(self,sentence1,sentence2):
        
        test_data = self._convert_examples_to_features(examples=self._create_examples(sentence1,sentence2),
                                                tokenizer=self.tokenizer,
                                                label_list=self.label_list,
                                                max_length=self.max_seq_length,
                                                # output_mode=output_mode,
                                                # pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=0)

        pred_sampler = SequentialSampler(test_data)
        pred_dataloader = DataLoader(test_data, sampler=pred_sampler, batch_size=self.pred_batch_size,
                                     collate_fn=self._collate_fn)

        for step, batch in enumerate(pred_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]
                          }          
                outputs = self.model(**inputs)
                _, logits = outputs[:2]
                prob = logits[0].softmax(-1).detach().cpu().numpy()
                predict = torch.max(logits[0].data, 0)[1].cpu().numpy()
                return prob[1]

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, label,input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"