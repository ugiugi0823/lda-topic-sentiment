import numpy as np
import pandas as pd

# 오늘 수집한 s&p 지수: 5.23~6.8
snp = [4145.58, 4115.24, 4151.28, 4205.45, 4205.52, 4205.52, 4205.52, 4205.52, 4179.83, 4221.02, 4282.37, 4279.37,
       4276.37, \
       4273.79, 4283.85, 4267.52, 4293.93, 4298.86, 4298.86, 4298.86, 4338.93, 4369.01, 4372.59, 4425.84]


#감저분석 결과 df 넣기
def get_sentiment_score(sentiment_df):
    import pandas as pd
    import numpy as np

    sentiment_df['tweetDate'] = sentiment_df['tweetDate'].apply(lambda x: x.split()[0])
    date_group = sentiment_df.groupby('tweetDate')
    new_df = pd.DataFrame(columns = ['date', 'agg_score'])

    for group in date_group:
        total_pos = 0
        total_neg = 0
        total_neu = 0

        pos_idx = 2
        neg_idx = 0
        neu_idx = 1

        date = group[0]
        df = group[1]
        total_tweet = df.shape[0]

    for i in df.iloc[:, -3:].values:
        idx = np.argsort(i)[-1]
        if idx == pos_idx:
          total_pos += 1
        if idx == neg_idx:
          total_neg += 1
        if idx == neu_idx:
          total_neu += 1

    result = ((total_pos - total_neg) / total_tweet) * (1 - (total_neu / total_tweet))
    new_df.loc[len(new_df) + 1] = [date, result] #df에 추가
    return pd.Series(data = new_df['agg_score']).rename(new_df['date'])


# 토픽 이름, 일별 감정점수 집계, 날짜
def get_correlation(df, index, window):
    import numpy as np
    from scipy import stats
    sentiment_score = get_sentiment_score(df)
    window = window + 1

    agg_date = [sentiment_score.index[i - 1] for i in range(window, len(sentiment_score) + 1)]
    agg_score = [sentiment_score[i - window : i].mean() for i in range(window, len(sentiment_score) + 1)]
    agg_index = [(index[i-1] - index[i-window])/index[i-window] for i in range(window, len(index) + 1)]

    if len(agg_date) < len(agg_index):
        agg_index = agg_index[:len(agg_date)]

    # 일단 지금 결과는 하나의 값으로 나오는데 그래프 그리려면 위의 sentiment_score 배열 자체도 반환해야 함.
    return pd.DataFrame(zip(agg_date,agg_score, agg_index), columns=['agg_date', 'agg_score', 'agg_index']), stats.pearsonr(agg_score, agg_index)


#코랩용 코드
'''for i in range(10):
    df = pd.read_csv(
        '/content/drive/MyDrive/Colab Notebooks/BERT/SA/sentiment_result/6.7_lda_sigmoid/' + 'topic_' + str(
            i) + '_sig_text.csv', lineterminator='\n')
    print(get_correlation(df, snp))'''
