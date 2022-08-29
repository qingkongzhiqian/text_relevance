#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
__file__    :   app.py
__time__    :   2022/07/22 17:41:57
__author__  :   yangning
__copyright__   :  Copyright 2022
'''

from flask import Flask, render_template, request, jsonify
from Text_relevance.model_predict import ModelPredict
app = Flask(__name__)

model_predict = ModelPredict()

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])

def index(name=None):
    return render_template('index.html', name = name)

@app.route('/relevance', methods=['GET', 'POST'])
def relevance():
    return render_template('relevance.html')

@app.route('/answer', methods=['GET', 'POST'])
def answer():

    sentence1 = request.args.get('sentence1')
    sentence2 = request.args.get('sentence2')
    pair = sentence1 + "||" + sentence2
    action = model_predict.model_predict_action(pair)
    model_predict.logging_message(sentence1,sentence2,action)
    return jsonify([action])

@app.route('/get_all_relation', methods=['GET', 'POST'])
def get_all_relation():
    return render_template('all_relation.html')

if __name__ == '__main__':
    app.debug=True
    app.run(host="0.0.0.0",port=58088,debug=True)
    # app.run(port=18080)
