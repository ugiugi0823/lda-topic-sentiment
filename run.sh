pip install transformers
git clone https://github.com/ugiugi0823/lda_topic_sentiment.git
cd /content/lda_topic_sentiment
gdown '1-nX8JfYDyZEo7fmI5SRhnBbhaPQ-dRl6&confirm=t'
unzip lda_6_8.zip
# lda_6_7_total.pickle
gdown '1_zn7_5TB6ZngKqWjp6uaKWGMgydHSHhw&confirm=t'
# lda_6_9.pickle
gdown '1IdegrqkFzPUXLskVAOc5VPN65sGyQyen&confirm=t'

python main.py --ldaa 'lda_6_9.pickle' \
  --db 'url_check_6_8.db' \
  --n_topic 10 \
  --sentiment "cardiffnlp/twitter-roberta-base-sentiment-latest" \
  --output_dir '/content/drive/MyDrive/sw/project/LDA/output/' \
  --topic_text_dir '/content/drive/MyDrive/sw/project/LDA/topic_text/' \
  --topic_dir '/content/drive/MyDrive/sw/project/LDA/result/' 
