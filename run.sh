# 최종 날짜 2023.06.18, 반영한 DB
pip install transformers
cd /content/lda_topic_sentiment
gdown '1wdTqzP202eQLirHApsybI8bIiROBU_H-&confirm=t'
unzip pickle_db_file.zip -d ./file/

rm -r cardiffnlp
python main.py --lda 'lda_6_9.pickle' \
  --db 'url_check_6_8.db' \
  --n_topic 10 \
  --sentiment "cardiffnlp/twitter-roberta-base-sentiment-latest" \
  --topic_text_dir './doc_topic/' \
  --topic_dir './sentiment_result/' 
