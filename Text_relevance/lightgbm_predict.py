#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
__file__    :   lightgbm_predict.py
__time__    :   2022/06/22 15:14:59
__author__  :   yangning
__copyright__   :  Copyright 2022
'''

import sys
import lightgbm as lgb
import numpy as np

class LightgbmPredict:
    def __init__(self):

        self.os_path = sys.path[0]
        self.model_name = "lightgbm_model"
        self.model_path = self.os_path + "/Text_relevance/" + self.model_name
        self.gbm = lgb.Booster(model_file=self.model_path + "/" + "gbm_model")

    def predict(self,bert_theseus,textrcnn):

        pred = self.gbm.predict(np.array([[bert_theseus, textrcnn]]))
        return round(pred[0][1],4)
