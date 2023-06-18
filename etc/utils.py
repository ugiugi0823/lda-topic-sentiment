#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import sqlite3
import shutil

import numpy as np
from etc.preproc import replaceURL, removeAtUser, removeHashtagInFrontOfWord

def get_db(args):

  # # db 경로 자동화
  # file_list = os.listdir(os.path.join(os.getcwd(), 'file'))
  # # .db 확장자를 가진 파일 이름 찾기
  # db_files = [filename for filename in file_list if filename.endswith('.db')]  
  # db_file = db_files[0]
  



  # # SQLite3 연결
  # conn = sqlite3.connect(f'./file/{db_file}')

  # # 쿼리 실행 및 데이터프레임 생성
  # # query = 'select * from tweet where hasURL=0;'
  # # ex = pd.read_sql_query(query, conn)

  # # 연결 종료
  # conn.close()
  dbb = args.db
  ex = pd.read_csv(f'./data/{dbb}')

  ex = ex[(ex['hasURL'] == 0.0)]
  
  raw = ex[['id', 'tweetDate', 'rawContent', 'preproc']]
  print('총 트윗 개수 ',len(raw))
  drop = len(raw) - len(raw.dropna())
  print(f'{drop}개의 중복 문서를 제거 했습니다.')
  raw = raw.dropna()
  missing_values = raw.isnull().sum()
  print('결측치가 확실하게 없는지 확인, 0이면 없는 것! ',missing_values)
  
  doc = raw.rawContent.tolist()

  return doc, raw



def setup(args):
  model_name = args.sentiment
  folder_name = model_name.split('/')[0]
  

  assert 'drive' in os.listdir('/content') # 당황하지 마세요! 드라이브 연결을 안해놓았어요! 코랩 드라이브 연결해주세요!
  print('구글 Drive 환경에 폴더를 제작합니다.')
  if args.drive:
    os.makedirs('/content/drive/MyDrive/inisw08', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/inisw08/lda-topic-sentiment', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/inisw08/lda-topic-sentiment/sentiment_result', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/inisw08/lda-topic-sentiment/doc_topic', exist_ok=True)

  
  else:
    print('로컬 환경에 폴더를 제작합니다.')
    os.makedirs('/content/inisw08', exist_ok=True)
    os.makedirs('/content/inisw08/lda-topic-sentiment', exist_ok=True)
    os.makedirs('/content/inisw08/lda-topic-sentiment/sentiment_result', exist_ok=True)
    os.makedirs('/content/inisw08/lda-topic-sentiment/doc_topic', exist_ok=True)


  if os.path.exists(folder_name):
    # 폴더 삭제
    # os.rmdir(folder_name)
    shutil.rmtree(folder_name)
    print("폴더 삭제 완료. 설치를 시작합니다.!")
  else:
    print("해당 모델 폴더가 존재하지 않습니다. 설치를 시작합니다.!")




preproc = []
def get_preproc(doc):
  for text in doc:
    text = replaceURL(text)
    text = removeAtUser(text)
    text = removeHashtagInFrontOfWord(text)
    preproc.append(text)

  return pd.Series(preproc)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

