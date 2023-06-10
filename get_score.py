import pandas as pd
from etc.correlation import get_correlation

snp = [4145.58, 4115.24, 4151.28, 4205.45, 4205.52, 4205.52, 4205.52, 4205.52, 4179.83, 4221.02, 4282.37, 4279.37,
       4276.37, \
       4273.79, 4283.85, 4267.52, 4293.93]


def get_score():
  for i in range(10):
      df = pd.read_csv(
          './result/' + 'result_topic_' + str(
              i) + '_text.csv', lineterminator='\n')
      print(f'{i}번쨰_______________________________________')
      print(get_correlation(df, snp))
      # print(type(a))
      # print(type(b))
