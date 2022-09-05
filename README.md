# text_relevance
### 项目描述

基于flask搭建了一个web端的文本相关性计算服务，需要GPU启动。

详细方案见：https://zhuanlan.zhihu.com/p/561424099

### 运行环境

```python
#运行环境：environment.yml
python app.py
```

### 目录结构

~~~python

text_relevance
├─ Text_relevance
│  ├─ BERT-THESEUS #BERT-THESEUS模型参数文件夹
│  │  ├─ config.json
│  │  ├─ pytorch_model.bin
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ Bert_predict.py
│  ├─ Bert_theseus_predict.py
│  ├─ Glyph #字形离线特征表
│  │  ├─ bihuashu_2w.txt
│  │  ├─ hanzijiegou_2w.txt
│  │  ├─ pianpangbushou_2w.txt
│  │  └─ sijiaobianma_2w.txt
│  ├─ TextRCNN #TextRCNN模型参数文件夹
│  │  ├─ data
│  │  │  ├─ embedding_SougouNews.npz
│  │  │  └─ vocab.pkl
│  │  └─ saved_dict
│  │     └─ TextRCNN.ckpt
│  ├─ TextRCNN_predict.py
│  ├─ bert_of_theseus
│  │  ├─ __init__.py
│  │  ├─ modeling_bert_of_theseus.py
│  │  └─ replacement_scheduler.py
│  ├─ glyph_predict.py
│  ├─ levenshtein_predict.py
│  ├─ lightgbm_model #lightgbm模型参数文件夹
│  │  └─ gbm_model
│  ├─ lightgbm_predict.py
│  └─ model_predict.py
├─ app.py #模型启动文件
├─ logs
│  └─ log.txt
```
~~~

### 示例

```python
sentence1:谁知道这是什么牌子的包？
sentence2:大家知道这是什么牌子的粉吗？

>>>
Bert_theseus predict relevance :1e-04
Textrcnn predict relevance :0.0055
Glyph predict relevance :0.8077
Levenshtein predict relevance :0.7692
****************************************************************************************************
Final semantics relevance :0.0021
Final glyph relevance :0.7885
```

## 模型分享
|      模型      | 百度网盘                                                     | 模型描述                     |
| :------------: | ------------------------------------------------------------ | ---------------------------- |
|  BERT-THESEUS  | 链接: https://pan.baidu.com/s/1Ty5MWdG7VqKPsOP8xUn3bw?pwd=uzyi 提取码: uzyi | BERT-THESEUS模型参数文件夹   |
|    TextRCNN    | 链接: https://pan.baidu.com/s/1Ty5MWdG7VqKPsOP8xUn3bw?pwd=uzyi 提取码: uzyi | TextRCNN模型参数文件夹       |
| lightgbm_model | 链接: https://pan.baidu.com/s/1Ty5MWdG7VqKPsOP8xUn3bw?pwd=uzyi 提取码: uzyi | lightgbm_model模型参数文件夹 |



