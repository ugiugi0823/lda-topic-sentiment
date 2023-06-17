# 최종 날짜 2023.06.18, 반영한 DB
pip install transformers
cd /content/lda_topic_sentiment

rm -r cardiffnlp
python main.py --lda 'lda_6_13.pickle' \
  --db 'preproc_6_2.csv' \
  --n_topic 10 \
  --sentiment "cardiffnlp/twitter-roberta-base-sentiment-latest" \
  --topic_text_dir './doc_topic/' \
  --topic_dir './sentiment_result/' 
