#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import argparse
import time
from datetime import datetime
import gc
import pickle
import os

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import urllib.request

import pandas as pd
from lda import get_topic_text
from etc.utils import setup, sigmoid
from etc.correlation import get_correlation






def sentiment(args):
  # args
  n_topic = args.n_topic
  MODEL = args.sentiment
  


  # tokenizer, model
  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  model.save_pretrained(MODEL)
  print(f'main, {MODEL} 모델 로드 완료~')

  # cuda empty
  torch.cuda.empty_cache()

  if torch.cuda.device_count() > 1:
    print('Multi-GPU로 병렬화')
    model = nn.DataParallel(model)

  max_length = 512
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  model.to(device)
  # Multi-GPU로 병렬화

    
  model.eval()
  current_time = datetime.now()
  
  print("시작 시간!!!", current_time)
  
  
  topic_box = []
  for n in range(n_topic):
    print(f'{len(topic_box)} 번째 Topic Sentiment Analysis ')
    gc.collect()
    if args.drive:
      df = pd.read_csv(f'/content/drive/MyDrive/inisw08/lda-topic-sentiment/doc_topic/topic_{n}_text.csv',  lineterminator='\n')
    else:
      df = pd.read_csv(f'/content/inisw08/lda-topic-sentiment/doc_topic/topic_{n}_text.csv',  lineterminator='\n')
    

    texts = df.text.tolist()
    print(str(n) + '번째 토픽 그리고 길이'+str(len(texts)))
    topic = []
    
    start_time_2 = datetime.now()
    for text in texts:
      
      encoded_input = tokenizer(text, max_length=max_length, return_tensors='pt', padding=True, truncation=True)
      encoded_input = encoded_input.to(device)

      with torch.no_grad():
        output = model(**encoded_input)
      
      scores = output[0][0].detach().cpu().numpy()
      scores = softmax(scores)
      

      topic.append(scores)
      if len(topic) % 10000 == 0:
        print(scores)
        print(f'{len(topic)}번째 트윗 검사중~~!!')
 
    
    end_time_2 = datetime.now()
    execution_time = end_time_2 - start_time_2

    print(f"실행 시간: {execution_time}")  
    topic_box.append(topic)
    topic_v = np.array(topic)
    df_v = df.values
    
    combined_array_2 = np.concatenate((df_v, topic_v), axis=1)
    df_Re2 = pd.DataFrame(combined_array_2)
    df_Re2.columns = ['topic', 'id', 'tweetDate', 'text', 'negative', 'neutral', 'positive']

    if args.drive:
      df_Re2.to_csv(f'/content/drive/MyDrive/inisw08/lda-topic-sentiment/sentiment_result/result_topic_{n}_text.csv', index=False)    
    else:
      df_Re2.to_csv(f'/content/inisw08/lda-topic-sentiment/sentiment_result/result_topic_{n}_text.csv', index=False)    
    print(f'{n}번째 토픽 정리 완료')

  end_time = datetime.now()
  print("종료 시간 :", end_time)
  return print('종료 되었습니다!')

