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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def main(args):
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
  
  
  box = []
  for n in range(n_topic):
    print(f'{len(box)} 번째 Topic Sentiment Analysis ')
    gc.collect()
    df = pd.read_csv(f'{topic_text_dir}/topic_{n}_text.csv',  lineterminator='\n')
    

    texts = df.text.tolist()
    print(str(n) + '번째 토픽 그리고 길이'+str(len(texts)))
    sss = []
    sig = []
    soft = []
    start_time_2 = datetime.now()
    for text in texts:
      
      encoded_input = tokenizer(text, max_length=max_length, return_tensors='pt', padding=True, truncation=True)
      encoded_input = encoded_input.to(device)

      with torch.no_grad():
        output = model(**encoded_input)
      
      scores = output[0][0].detach().cpu().numpy()
      scores = softmax(scores)
      

      
      # scores_sig = sigmoid(scores)
      # sig.append(scores_sig)
      # soft.append(scores)

      # ranking = np.argsort(scores)
      # ranking = ranking[::-1]
      # ss =[]
      # for i in range(scores.shape[0]):
      #   s = scores[ranking[i]]
      #   s = np.round(s * 100, 2)
      #   ss.append(s)
      #   # print(s)
      sss.append(scores)
      if len(sss) % 1000 == 0:
        print(scores)
        print(f'{len(sss)}번째 트윗 검사중~~!!')
 
    
    end_time_2 = datetime.now()
    execution_time = end_time_2 - start_time_2

    print(f"실행 시간: {execution_time}")  
    # print("실행 시간: {:.2f} 초".format(execution_time))
    box.append(sss)
    sss_v = np.array(sss)
    df_v = df.values
    
    combined_array_2 = np.concatenate((df_v, sss_v), axis=1)
    df_Re2 = pd.DataFrame(combined_array_2)
    df_Re2.columns = ['topic', 'id', 'tweetDate', 'text', 'negative', 'neutral', 'positive']
    df_Re2.to_csv(f'{topic_dir}/topic_{n}_text.csv', index=False)    
    print(f'{n}번째 토픽 정리 완료')

  end_time = datetime.now()
  print("종료 시간 :", end_time)

  # 총 데이터 저장!
  with open(f'{output_dir}/output2.pkl', 'wb') as f:
    pickle.dump(box, f)
  return print('종료 되었습니다!')


if __name__ == '__main__':

  p = argparse.ArgumentParser()
  p.add_argument("--ldaa", default=".", help="lda_model 경로를 넣어주세요!")
  p.add_argument("--db", default=".", help="db file 경로를 넣어주세요!")
  p.add_argument("--sentiment", default="cardiffnlp/twitter-roberta-base-sentiment-latest", help="lda_model 경로를 넣어주세요!")
  p.add_argument("--n_topic", type=int , default=10, help="몇개의 토픽인지 적어주세요!")
  p.add_argument("--topic_dir", default="./lda_topic_sentiment/result", help="output 경로를 넣어주세요!")
  p.add_argument("--output_dir", default="./lda_topic_sentiment/output", help="output 경로를 넣어주세요!")
  p.add_argument("--topic_text_dir", default="./lda_topic_sentiment/topic_text", help="output 경로를 넣어주세요!")

  args = p.parse_args()
  setup(args)
  get_topic_text(args)
  main(args)



