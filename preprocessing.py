import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import re


def remove_stopwords(word):
    if word == "" or word is None:
        return word

    englishStopwords = stopwords.words('english')
    text_nostop = []
    if word not in englishStopwords:
        return word.lower()


def tokenize(text):
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
    text = tokenizer.tokenize(text)
    return text


def remove_url(word):
    if word == "" or word is None:
        return word

    text_filtered = []
    regex = re.compile("^https?:\/\/.*[\r\n]*")
    if not regex.match(word):
        return word


def remove_numbers(word):
    if word == "" or word is None:
        return word
    return re.sub(r'\d+', '', word)

def remove_puncuation(word):
    if word == "" or word is None:
        return word
    # Only allowing words, question mark or exclamation
    pattern_question_mark = r'\?'
    pattern_exclamation_mark = r'\!'
    pattern_word = r'\w+'
    pattern_hashtag = r'#\w+'

    filtered_text = []


    if re.match(pattern_word,word) or re.match(pattern_exclamation_mark,word) or re.match(pattern_question_mark,word) or re.match(pattern_hashtag,word):
        return word


def lemmatise(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)
