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
  p.add_argument("--drive", action='store_true', help="Drive 저장하고 싶으면")

  args = p.parse_args()
  setup(args)
  get_topic_text(args)
  sentiment(args)
  #get_score()

  



