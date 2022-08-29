#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
__file__    :   glyph_predict.py
__time__    :   2022/06/06 15:12:57
__author__  :   yangning
__copyright__   :  Copyright 2022
'''
import sys

class GlyphPredict:
    def __init__(self):

        self.os_path = sys.path[0]
        self.model_name = 'Glyph'
        self.bihuashuDict = self._initDict(self.os_path + '/Text_relevance/' + self.model_name + '/bihuashu_2w.txt')
        self.hanzijiegouDict = self._initDict(self.os_path + '/Text_relevance/' + self.model_name + '/hanzijiegou_2w.txt')
        self.pianpangbushouDict = self._initDict(self.os_path + '/Text_relevance/' + self.model_name + '/pianpangbushou_2w.txt')
        self.sijiaobianmaDict = self._initDict(self.os_path + '/Text_relevance/' + self.model_name +'/sijiaobianma_2w.txt')

        self.hanzijiegouRate = 10
        self.sijiaobianmaRate = 8
        self.pianpangbushouRate = 6
        self.bihuashuRate = 2

    def _initDict(self,path):
        dict = {}; 
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                # 移除换行符，并且根据空格拆分
                splits = line.strip('\n').split(' ')
                key = splits[0]
                value = splits[1]
                dict[key] = value
        return dict

    def _bihuashuSimilar(self,charOne, charTwo): 
        valueOne = self.bihuashuDict.get(charOne,-1)
        valueTwo = self.bihuashuDict.get(charTwo,-1)        
        if valueOne == valueTwo == -1:return 0.0

        numOne = int(valueOne)
        numTwo = int(valueTwo)
        alpha = 1e-5
        
        diffVal = 1 - abs((numOne - numTwo) / (max(numOne, numTwo) + alpha))
        return self.bihuashuRate * diffVal * 1.0;

    def _hanzijiegouSimilar(self,charOne, charTwo): 
        valueOne = self.hanzijiegouDict.get(charOne,-1)
        valueTwo = self.hanzijiegouDict.get(charTwo,-1)
        
        if valueOne == valueTwo and valueOne != -1:
            # 后续可以优化为相近的结构
            return self.hanzijiegouRate * 1
        return 0

    def _sijiaobianmaSimilar(self,charOne, charTwo): 
        valueOne = self.sijiaobianmaDict.get(charOne,-1)
        valueTwo = self.sijiaobianmaDict.get(charTwo,-1)
        if valueOne == valueTwo == -1:return 0.0

        totalScore = 0.0
        minLen = min(len(valueOne), len(valueTwo))
        
        for i in range(minLen):
            if valueOne[i] == valueTwo[i]:
                totalScore += 1.0
        
        totalScore = totalScore / minLen * 1.0
        return totalScore * self.sijiaobianmaRate

    def _pianpangbushoutSimilar(self,charOne, charTwo): 
        valueOne = self.pianpangbushouDict.get(charOne,-1)
        valueTwo = self.pianpangbushouDict.get(charTwo,-1)
        if valueOne == valueTwo == -1:return 0.0
        
        if valueOne == valueTwo:
            # 后续可以优化为字的拆分
            return self.pianpangbushouRate * 1
        return 0;    

    def _similar(self,charOne, charTwo):
        if charOne == charTwo:
            return 1.0
        
        sijiaoScore = self._sijiaobianmaSimilar(charOne, charTwo)
        jiegouScore = self._hanzijiegouSimilar(charOne, charTwo)
        bushouScore = self._pianpangbushoutSimilar(charOne, charTwo)
        bihuashuScore = self._bihuashuSimilar(charOne, charTwo)
        
        totalScore = sijiaoScore + jiegouScore + bushouScore + bihuashuScore
        totalRate = self.hanzijiegouRate + self.sijiaobianmaRate + self.pianpangbushouRate + self.bihuashuRate
        
        result = totalScore*1.0 / totalRate * 1.0
        return result

    def predict(self,sentence1,sentence2):

        try:
            match_score = []
            for a in sentence1:
                if not '\u4e00' <= a <= '\u9fff':continue
                score = 0.0
                for b in sentence2:
                    if not '\u4e00' <= b <= '\u9fff':continue
                    score = max(score,self._similar(a,b))
                match_score.append(score)  
            prob = sum(match_score) / len(sentence1)    
            return prob
        except:
            return 0.0    