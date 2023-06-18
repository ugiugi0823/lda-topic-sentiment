# 최종 날짜 2023.06.18, 반영한 DB
pip install transformers


gdown '1wVh8tP0XcOuabv9EUa6hnXiykqjTqQ1t&confirm=t'
unzip database_csv.zip -d /content/lda-topic-sentiment/data




rm -r cardiffnlp
python main.py \
  --lda 'lda_6_13.pickle' \
  --db 'preproc_6_15.csv' \
  --n_topic 10 \
  --sentiment "cardiffnlp/twitter-roberta-base-sentiment-latest" \
  --drive
