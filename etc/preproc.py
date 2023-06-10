import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words = stop_words | {'would', 'im', 'ive', 'thing', 'thats', 'probably', 'someone', 'think', 'actually', 'cant',
                           'much', 'anything', 'something', 'lol', 'stuff', 'bit', 'really', 'still', 'shit',
                           'literally', 'even', 'always', 'time', 'never', 'little', 'yeah', 'didnt', 'pretty',
                           'friend', 'nothing', 'maybe', 'look', 'lot', 'came', 'isnt', 'theyre', 'dont', 'doesnt',
                           'um', 'ye', 'hmm', 'un', 'uh', 'eh', 'huh', 'ya', 'yo', 'yea', 'ah', 'nah', 'fuck', 'u',
                           'uu',
                           'oh', 'fucking', 'apple', 'amazon', 'microsoft', 'google', 'nvidia', 'tesla'}


def preprocess(text: str):
    """리트윗, 특수문자, 숫자, #, @유저명, 구두점, 불용어 지우기 + 토큰화, 표제어추출 전부"""

    # remove unnecessary words
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'^rt[\s]+', '', text)
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = removeEmoticons(text)

    text = text.lower()

    # tokenizing and lemmatization
    tweet_tokens = word_tokenize(text)
    tweet_tokens = [t for t in tweet_tokens if t not in stop_words]
    if len(tweet_tokens) < 4:
        return None
    lemm_tokens = [lemmatizer.lemmatize(token) for token in tweet_tokens]
    return ' '.join(lemm_tokens)


def remove_waste(text):
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'^rt[\s]+', '', text)
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = removeEmoticons(text)
    return text


def removeUnicode(text):
    """ 특수문자 지우기 """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    return text


def replaceURL(text):
    """ url 지우기 """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    return text


def removeAtUser(text):
    """ @유저명 지우기 """
    text = re.sub('@[^\s]+', '', text)
    return text


def removeHashtagInFrontOfWord(text):
    """  # 지우기 """
    text = re.sub(r'#', r'', text)
    return text


def removeNumbers(text):
    """ 숫자 지우기 """
    text = re.sub(r'[0-9]', '', text)
    return text


def tokenizing_lemmatization(text):
    tweet_tokens = word_tokenize(text.lower())
    lemm_tokens = [lemmatizer.lemmatize(token) for token in tweet_tokens]
    return ' '.join(lemm_tokens)


def remove_stopwords(text):
    tweet_tokens = word_tokenize(text.lower())
    tweet_tokens = [t for t in tweet_tokens if t not in stop_words]
    return ' '.join(tweet_tokens)


def stopwords_lemma(text):
    tweet_tokens = word_tokenize(text.lower())
    tweet_tokens = [t for t in tweet_tokens if t not in stop_words]
    lemm_tokens = [lemmatizer.lemmatize(token) for token in tweet_tokens]
    return ' '.join(lemm_tokens)


def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(
        ':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:',
        '', text)
    return text

# print(preprocess('''In honor of
# @MarvelStudios
# ’ #GotGVol3 we’re giving away this never opened Microsoft Zune. We have no idea if it works. Like and RT for a chance to win! US 18+. Ends 5/17/23. Rules: https://msft.it/6002gX0tC. #ZuneSweepstakes'''))
