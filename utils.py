#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import sqlite3
import shutil

import numpy as np
from preproc import replaceURL, removeAtUser, removeHashtagInFrontOfWord

def get_db(text):
  # db 경로 자동화
  target_folder='file'
  if os.path.exists(os.path.join(os.getcwd(), target_folder)):
    print(f"현재 경로에 {target_folder} 폴더가 있습니다.")
  else:
    print(f"현재 경로에 {target_folder} 폴더가 없습니다.")

  file_list = os.listdir(os.path.join(os.getcwd(), 'file'))
  # .db 확장자를 가진 파일 이름 찾기
  db_files = [filename for filename in file_list if filename.endswith('.db')]  
  db_file = db_files[0]
  



  # SQLite3 연결
  conn = sqlite3.connect(db_file)

  # 쿼리 실행 및 데이터프레임 생성
  query = 'select * from tweet where hasURL=0;'
  ex = pd.read_sql_query(query, conn)

  # 연결 종료
  conn.close()
  raw = ex[['id', 'tweetDate', 'rawContent', 'preproc']]
  drop = len(raw) - len(raw.dropna())
  print(f'{drop}개의 중복 문서를 제거 했습니다.')
  raw = raw.dropna()
  doc = raw.rawContent.tolist()

  return doc, raw



def setup(args):
  model_name = args.sentiment
  folder_name = model_name.split('/')[0]
  os.makedirs("lda_model", exist_ok=True)

  os.makedirs("topic_text", exist_ok=True)
  os.makedirs("output", exist_ok=True)
  os.makedirs("result", exist_ok=True)

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

