#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import sqlite3

def get_db(text):
  # SQLite3 연결
  conn = sqlite3.connect(text)

  # 쿼리 실행 및 데이터프레임 생성
  query = 'select * from tweet where hasURL=0;'
  ex = pd.read_sql_query(query, conn)

  # 연결 종료
  conn.close()
  raw = ex[['id', 'tweetDate', 'rawContent', 'preproc']]
  drop = len(raw) - len(raw.dropna())
  print('중복 개수 ', drop)
  raw = raw.dropna()
  doc = raw.preproc

  return doc, raw



def setup(args):
  model_name = args.sentiment
  folder_name = model_name.split('/')[1]
  os.makedirs("topic_text", exist_ok=True)
  if os.path.exists(folder_name):
    # 폴더 삭제
    os.rmdir(folder_name)
    print("폴더 삭제 완료")
  else:
    print("해당 폴더가 존재하지 않습니다.")

