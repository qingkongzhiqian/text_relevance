#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
__file__    :   levenshtein_predict.py
__time__    :   2022/06/17 15:07:45
__author__  :   yangning
__copyright__   :  Copyright 2022
'''

import sys
import Levenshtein

class LevenshteinPredict(object):
    def __init__(self):
        pass
    
    def predict(self, sentence1, sentence2):
        
        prob = Levenshtein.ratio(sentence1,sentence2)
        return prob

