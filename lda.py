#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import pickle
import numpy as np
from etc.utils import get_db, get_preproc


 

def get_topic_text(args):
  # args
  lda_model = args.lda
  
  

  # 모델 경로 자동화
  target_folder='file'
  if os.path.exists(os.path.join(os.getcwd(), target_folder)):
    print(f"현재 경로에 {target_folder} 폴더가 있습니다.")
  else:
    print(f"현재 경로에 {target_folder} 폴더가 없습니다.")

  # .pickle 확장자를 가진 파일 이름 찾기
  file_list = os.listdir(os.path.join(os.getcwd(), 'file'))
  pickle_files = [filename for filename in file_list if filename.endswith('.pickle')]
  pickle_file = pickle_files[0]

  # numpy version check
  required_version = "1.24.2"
  current_version = np.__version__
  # assert current_version == required_version, f"NumPy 버전이 {required_version}이 아닙니다. 현재 버전: {current_version}"


  # lad model 
  with open(f'./file/{pickle_file}', 'rb') as f:
    lda = pickle.load(f) # 단 한줄씩 읽어옴

  doc, raw = get_db(args)
  print(f'lda, DB로부터 {len(doc)}개의 트윗 DataFrame 불러오기')
  doc = get_preproc(doc)
  print('시간이 조금 걸립니다!')
  print('lda, Data Preprocessing')
  


  # count_vec
  count_vect = CountVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
  fit_vect = count_vect.fit_transform(doc)

  # 문장 LDA 돌리기
  doc_topic = lda.transform(fit_vect)
  doc_topic = np.array(doc_topic)


  # LDA 후, 토픽별 문장 확률값
  topic_names = [str(i) for i in range(doc_topic.shape[1])]
  doc_topic_df = pd.DataFrame(data = doc_topic, columns = topic_names)


  # 토픽별로 문장 정리
  doc_topic_df_v = doc_topic_df.values

  max_column = doc_topic_df.idxmax(axis=1)
  max_column_v = max_column.values
  max_column_v = max_column_v.reshape(-1, 1)


  combined_array = np.concatenate((doc_topic_df_v, max_column_v), axis=1)
  df_Re = pd.DataFrame(combined_array)


  # topic
  topic_v = np.array(df_Re.iloc[:, -1])
  topic_v = topic_v.reshape(-1, 1)


  # text
  array_v = np.array(raw[['id', 'tweetDate', 'rawContent']])
  # array_v = array_v.reshape(-1, 1)


  com = np.concatenate((topic_v, array_v), axis=1)
  com_df = pd.DataFrame(com)
  com_df.columns = ['topic', 'id','tweetDate','text']


  df_sorted = com_df.sort_values(by='topic', ascending=True)
  # 반복문으로 파일 저장
  for i in range(10):
    if args.drive:
      df_sorted[df_sorted['topic'] == str(i)].to_csv(f'/content/drive/MyDrive/inisw08/lda-topic-sentiment/doc_topic/topic_{i}_text.csv', index=False)
    else:
      df_sorted[df_sorted['topic'] == str(i)].to_csv(f'/content/inisw08/lda-topic-sentiment/doc_topic/topic_{i}_text.csv', index=False)

  return print('lda, 토픽별 문장 분류 및 저장 완료')

