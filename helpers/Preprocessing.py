from Tweets import Tweets
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import re


# Removing stopword based off nltk english stopwords
def remove_stopwords(word):
    if word == "" or word is None:
        return word

    englishStopwords = stopwords.words('english')

    # If the word is not a stop word, return it
    if word not in englishStopwords:
        #return word.lower()
        return word


# Tokenizing the tweet text by using NLTK's TweetTokenizer which also lowercases words
def tokenize(text):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    text = tokenizer.tokenize(text)
    return text


# Removing urls
def remove_url(word):
    if word == "" or word is None:
        return word

    # Using regex to match urls
    regex = re.compile("^https?:\/\/.*[\r\n]*")
    if not regex.match(word):
        return word


# Removing numbers
def remove_numbers(word):
    if word == "" or word is None:
        return word

    # Replacing numbers with an empty space
    return re.sub(r'\d+', '', word)


# Removing puncuation
def remove_puncuation(word):
    if word == "" or word is None:
        return word

    # Only allowing words, question mark or exclamation marks
    pattern_question_mark = r'\?'
    pattern_exclamation_mark = r'\!'
    pattern_word = r'\w+'
    pattern_hashtag = r'#\w+'
    #pattern_underscore = r'\_'

    # If the word (token) is either a word or a question mark or an exclamation mark it is returned
    if re.match(pattern_word,word) or re.match(pattern_exclamation_mark,word) or re.match(pattern_question_mark,word) or re.match(pattern_hashtag,word):
        return word


def lemmatise(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def remove_underscores(word):
    if word == "" or word is None:
        return word

    tempReturn = []
    for char in word:
        if char != "_":
            tempReturn.append(char)
    return ''.join(tempReturn)

def preprocess(tweets,labels,location):
    tweets_object = Tweets()
    for i in range(0, len(tweets)):
        #tweets[i] = " ".join(tweets[i].split())
        tweets[i] = tokenize(tweets[i])

        newText = []
        for word in tweets[i]:
            # Checking for @ Location
            # If the location is required the @ symbol is eliminated are the rest of the text is processed
            # If the location is not required the @ symbol and the remaining test are discarded
            if word == "@":
                if location:
                    continue
                if not location:
                    break

            word = lemmatise(word)
            word = remove_underscores(word)
            word = remove_stopwords(word)
            word = remove_url(word)
            word = remove_numbers(word)
            word = remove_puncuation(word)

            if word is not None and word != "":
                word = word.lower()

            if word is not None and word != "" and word == "___":
                continue

            if word is not None and word != "" and word[0] == "#":
                word = word[1:]

            if word is not None and word != "":
                newText.append(word)

        if len(newText) == 0:
            continue

        tweets_object.tweetsText.append(' '.join(newText))
        tweets_object.tweetsLabel.append(labels[i])
    return tweets_object