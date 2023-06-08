#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import pickle
import numpy as np
from utils import get_db, get_preproc


 

def get_topic_text(args):
  # lad model
  lda_model =args.lda_model
  with open(lda_model, 'rb') as f:
    lda = pickle.load(f) # 단 한줄씩 읽어옴

  doc, raw = get_db('url_check_6_6.db')
  doc = get_preproc(doc)


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
  array_v = np.array(raw.rawContent)
  array_v = array_v.reshape(-1, 1)


  com = np.concatenate((topic_v, array_v), axis=1)
  com_df = pd.DataFrame(com)
  com_df.columns = ['topic', 'text']


  df_sorted = com_df.sort_values(by='topic', ascending=True)
  # 반복문으로 파일 저장
  for i in range(10):
      df_sorted[df_sorted['topic'] == str(i)].to_csv(f'./topic_text/topic_{i}_text.csv', index=False)
  return print('토빅별 문장 나누기 후 저장 완료')

