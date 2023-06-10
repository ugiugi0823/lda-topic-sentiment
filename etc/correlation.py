import numpy as np
import pandas as pd

# 오늘 수집한 s&p 지수: 5.23~6.8
snp = [4145.58, 4115.24, 4151.28, 4205.45, 4205.52, 4205.52, 4205.52, 4205.52, 4179.83, 4221.02, 4282.37, 4279.37,
       4276.37, \
       4273.79, 4283.85, 4267.52, 4293.93]


#감저분석 결과 df 넣기
def get_sentiment_score(sentiment_df):
    sentiment_df['tweetDate'] = sentiment_df['tweetDate'].apply(lambda x: x.split()[0])

    # 긍정 부정 중립 중 가장 높은 점수를 해당 감성으로 치고 그 점수를 가져오는 함수
    # 이건 지금 중립 스코어까지 넣어버림.
    def highest(series):
        import numpy as np
        idx = np.argsort(series)[-1]
        result = series[idx]
        # 부정인 경우 -값으로 바꾼다.
        if idx == 2:
            result = -result
        return result

    sentiment_df['agg_score'] = sentiment_df.iloc[:, -3:].apply(highest, axis=1)
    sentiment_score = sentiment_df.groupby('tweetDate').agg(sum)['agg_score'] / sentiment_df.groupby('tweetDate').size()
    return sentiment_score


# 토픽 이름, 일별 감정점수 집계, 날짜
def get_correlation(df, snp, x_days=1):
    import numpy as np
    from scipy import stats
    sentiment_score = get_sentiment_score(df)

    # 일단 지금 결과는 하나의 값으로 나오는데 그래프 그리려면 위의 sentiment_score 배열 자체도 반환해야 함.
    return sentiment_score, stats.pearsonr(sentiment_score, snp)


#코랩용 코드
'''for i in range(10):
    df = pd.read_csv(
        '/content/drive/MyDrive/Colab Notebooks/BERT/SA/sentiment_result/6.7_lda_sigmoid/' + 'topic_' + str(
            i) + '_sig_text.csv', lineterminator='\n')
    print(get_correlation(df, snp))'''
