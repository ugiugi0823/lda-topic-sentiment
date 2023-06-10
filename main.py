#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import argparse

from lda import get_topic_text
from etc.utils import setup
from sentiment import sentiment
from get_score import get_score





if __name__ == '__main__':

  p = argparse.ArgumentParser()
  p.add_argument("--lda", default=".", help="lda_model 경로를 넣어주세요!")
  p.add_argument("--db", default=".", help="db file 경로를 넣어주세요!")
  p.add_argument("--sentiment", default="cardiffnlp/twitter-roberta-base-sentiment-latest", help="lda_model 경로를 넣어주세요!")
  p.add_argument("--n_topic", type=int , default=10, help="몇개의 토픽인지 적어주세요!")
  p.add_argument("--topic_dir", default="./lda_topic_sentiment/result", help="토픽별 감성분석 결과를 담는 경로")
  p.add_argument("--topic_text_dir", default="./lda_topic_sentiment/topic_text", help="lda 후 나온 토픽별 csv 경로")

  args = p.parse_args()
  setup(args)
  get_topic_text(args)
  sentiment(args)
  get_score()
  



