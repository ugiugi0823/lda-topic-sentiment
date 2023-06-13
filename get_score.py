import pandas as pd
from etc.correlation import get_correlation

snp = [4145.58, 4115.24, 4151.28, 4205.45, 4205.52, 4205.52, 4205.52, 4205.52, 4179.83, 4221.02, 4282.37, 4279.37,
       4276.37, \
       4273.79, 4283.85, 4267.52, 4293.93]


def get_score():
  for window_size in range(1, 6):
    print(f'window size: {window_size}')
    for i in range(10):
        df = pd.read_csv(
            './sentiment_result/' + 'result_topic_' + str(
                i) + '_text.csv', lineterminator='\n')
        print(f'토픽: {i}_______________________________________')
        #수정된 결과: 윈도우 사이즈 별로 집계된 df와 상관계수가 토픽별로 나옴
        print(get_correlation(df, snp, i+1))
        # print(type(a))
        # print(type(b))
