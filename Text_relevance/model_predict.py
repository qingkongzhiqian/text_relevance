#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
__file__    :   model_predict.py
__time__    :   2022/05/30 15:47:35
__author__  :   yangning
__copyright__   :  Copyright 2022
'''

import sys
import datetime
from Text_relevance.TextRCNN_predict import TextRCNNPredict
# from Text_relevance.Bert_predict import BertPredict
from Text_relevance.glyph_predict import GlyphPredict
from Text_relevance.levenshtein_predict import LevenshteinPredict
from Text_relevance.Bert_theseus_predict import BertTheseusPredict
from Text_relevance.lightgbm_predict import LightgbmPredict

class ModelPredict:

    def __init__(self):
        self.os_path = sys.path[0]
        self.textrcnn_predict = TextRCNNPredict()
        # self.bert_predict = BertPredict()
        self.bert_theseus_predict =BertTheseusPredict()
        self.glyph_predict = GlyphPredict()
        self.levenshtein_predict = LevenshteinPredict()
        self.lightgbm_predict = LightgbmPredict()
        self.log_dir = self.os_path + '/logs/' + 'log.txt' 

    def _TextRCNN_predict(self,sentence1,sentence2):
        return self.textrcnn_predict.predict(sentence1,sentence2)

    def _Bert_predict(self,sentence1,sentence2):
        return self.bert_predict.predict(sentence1,sentence2)

    def _Glyph_predict(self,sentence1,sentence2):
        return self.glyph_predict.predict(sentence1,sentence2)   

    def _Levenshtein_predict(self,sentence1,sentence2):
        return self.levenshtein_predict.predict(sentence1,sentence2)    

    def _Bert_Theseus_predict(self,sentence1,sentence2):     
        return self.bert_theseus_predict.predict(sentence1,sentence2)

    def _Lightgbm_predict(self,bert_theseus_pred,textrcnn_pred):
        return self.lightgbm_predict.predict(bert_theseus_pred,textrcnn_pred) 

    def _return_message(self,type,prob):
        hash_table = {
            'bert' : "Bert predict relevance :" ,
            'bert_theseus' : "Bert_theseus predict relevance :" ,
            'textrcnn' : "Textrcnn predict relevance :" ,
            'levenshtein' : "Levenshtein predict relevance :" ,
            'glyph' : "Glyph predict relevance :" ,
        }    

        return hash_table[type] + str(round(prob,4))

    def model_predict_action(self,question):

        try:
            sentence1,sentence2 = question.split("||")
            textrcnn = self._TextRCNN_predict(sentence1,sentence2)
            glyph = self._Glyph_predict(sentence1,sentence2)
            levenshtein = self._Levenshtein_predict(sentence1,sentence2)
            bert_theseus = self._Bert_Theseus_predict(sentence1,sentence2)

            lightgbm = self._Lightgbm_predict(bert_theseus,textrcnn)
            return_feature_message = self._return_message('bert_theseus',bert_theseus) + "<br/>" + \
                                self._return_message('textrcnn',textrcnn)+ "<br/>" + \
                                self._return_message('glyph',glyph) + "<br/>" + \
                                self._return_message('levenshtein',levenshtein) + "<br/>"

            return_ensemble_message = "*" * 100 + "<br/>" + \
                                    "Final semantics relevance :" + str(lightgbm) + "<br/>" + \
                                    "Final glyph relevance :" + str(round(sum([levenshtein+glyph]) / 2,4)) 

            return return_feature_message + return_ensemble_message

        except:
            return "Data format error ! please check your input" 

    def _logging_message_save(self,sentence1,sentence2,predict_message):

        with open(self.log_dir, 'a',encoding="utf-8") as f:
            f.write("****************** current time : {} ***********************".format(datetime.datetime.now()) + '\n')
            f.write("sentence1 : " + sentence1 + '\n')
            f.write("sentence2 : " + sentence2 + '\n')
            message = predict_message.split('<br/>')
            for item in message:
                if len(list(set(item))) == 1:continue
                f.write(item + '\n')

    def logging_message(self,sentence1,sentence2,predict_message):
        self._logging_message_save(sentence1,sentence2,predict_message)
