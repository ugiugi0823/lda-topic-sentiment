import pandas as pd
from correlation import get_correlation
def get_score():
  for i in range(10):
      df = pd.read_csv(
          '/content/lda_topic_sentiment/dd/' + 'topic_' + str(
              i) + '_text.csv', lineterminator='\n')
      print(f'{i}번쨰_______________________________________')
      print(get_correlation(df, snp))
      # print(type(a))
      # print(type(b))
