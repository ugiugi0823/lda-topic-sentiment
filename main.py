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
from utils import setup



def main(args):
  n_topic = args.n_topic
  MODEL = args.sentiment
  tokenizer = AutoTokenizer.from_pretrained(MODEL)


  # PT
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  model.save_pretrained(MODEL)


  torch.cuda.empty_cache()

  max_length = 512
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  current_time = datetime.now()
  
  print("시작 시간!!!", current_time)
  
  
  
  lists = [ [] for _ in range(10) ]

  for box in lists:
    gc.collect()
    
    for n in range(n_topic):
      df = pd.read_csv(f'./topic_text/topic_{n}_text.csv')
      texts = df.text.tolist()
      print(str(n) + '번째 토픽 그리고 길이'+str(len(texts)))
      for text in texts:
        start_time_2 = datetime.now()

        
        start_time = time.time()
        encoded_input = tokenizer(text, max_length=max_length, return_tensors='pt', padding=True, truncation=True)
        encoded_input = encoded_input.to(device)

        with torch.no_grad():
          output = model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        ss = []
        for i in range(scores.shape[0]):
          s = scores[ranking[i]]
          s = np.round(s * 100, 2)
          ss.append(s)
          # print(s)
        box.append(ss)
        if len(box) % 1000 == 0:
          print(f'{len(box)}번째 트윗 검사중~~!!')
        if len(box) % 10000 == 0:
          end_time_2 = datetime.now()
          execution_time = end_time_2 - start_time_2
          print(f"실행 시간: {execution_time.total_seconds()}초")
    
    end_time = datetime.now()
    print("종료 시간 :", end_time)

  # 저장할 리스트
  with open('output.pkl', 'wb') as f:
    pickle.dump(lists, f)
  return lists


if __name__ == '__main__':

  p = argparse.ArgumentParser()
  p.add_argument("--lda-model", default=".", help="lda_model 경로를 넣어주세요!")
  p.add_argument("--db", default=".", help="db file 경로를 넣어주세요!")
  p.add_argument("--sentiment", default="cardiffnlp/twitter-roberta-base-sentiment-latest", help="lda_model 경로를 넣어주세요!")
  p.add_argument("--n_topic", type=int , default=10, help="몇개의 토픽인지 적어주세요!")

  args = p.parse_args()
  setup(args)
  get_topic_text(args)
  main(args)



