#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
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
from utils import setup, sigmoid






def sentiment(args):
  # args
  n_topic = args.n_topic
  MODEL = args.sentiment
  output_dir = args.output_dir
  topic_dir = args.topic_dir
  topic_text_dir = args.topic_text_dir


  # tokenizer, model
  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  model.save_pretrained(MODEL)
  print(f'main, {MODEL} 모델 로드 완료~')

  # cuda empty
  torch.cuda.empty_cache()



  max_length = 512
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  model.to(device)
  model.eval()
  current_time = datetime.now()
  
  print("시작 시간!!!", current_time)
  
  
  topic_box = []
  for n in range(n_topic):
    print(f'{len(topic_box)} 번째 Topic Sentiment Analysis ')
    gc.collect()
    df = pd.read_csv(f'{topic_text_dir}/topic_{n}_text.csv',  lineterminator='\n')
    

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
      if len(topic) % 1000 == 0:
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
    df_Re2.to_csv(f'{topic_dir}/topic_{n}_text.csv', index=False)    
    print(f'{n}번째 토픽 정리 완료')

  end_time = datetime.now()
  print("종료 시간 :", end_time)

  # 총 데이터 저장!
  with open(f'{output_dir}/output2.pkl', 'wb') as f:
    pickle.dump(topic_box, f)
  return print('종료 되었습니다!')

