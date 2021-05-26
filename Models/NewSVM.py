import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from DataRetrival import get_train_data, get_test_data
from nltk import ngrams
from sklearn.svm import LinearSVC
from Scorer import *
"""
Notes from paper
- One vs rest
- SVM parameter C = 0.1
- bag of n-grams by words (of size 4)
- weighted by sub-linear TF-IDF
- minimum document frequency of 2

"""

#Things got a little festive at the office #christmas2016 @ RedRock…
#things got little festive office christmas redrock

#https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
def flatten_list(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def get_ngram(text, min, max):
    output = []
    for n in range (min ,max+1):
        ngramOutput = list(ngrams(text,n))
        newngramOutput = []
        for ngram in ngramOutput:
            newngramOutput.append('|'.join(ngram))
        if len(newngramOutput) !=0:
            output.append(newngramOutput)
    return flatten_list(output)

def get_word_ngrams(tweet_text, n_min,n_max):
    return get_ngram(tweet_text.split(),n_min,n_max)

def get_character_ngrams(tweet_text, n_min,n_max):
    character_ngram = []
    for word in tweet_text.split():
        character_ngram.append(get_ngram(list(word),n_min,n_max))

    return flatten_list(character_ngram)

def convert_tweets_to_ngrams(tweets_text, w_ngram_min,w_ngram_max,c_ngram_min,c_ngram_max):
    features =[]

    for tweet_text in tweets_text:
        tweetFeatures = []
        tweetFeatures.append(get_word_ngrams(tweet_text,w_ngram_min,w_ngram_max))
        tweetFeatures.append(get_character_ngrams(tweet_text,c_ngram_min,c_ngram_max))
        features.append(flatten_list(tweetFeatures))
    return features

def identity(x):
    return x

def run(train_tweets, test_tweets):
    w_ngram_min = 1
    w_ngram_max = 4
    c_ngram_min = 1
    c_ngram_max = 6


    train_features = convert_tweets_to_ngrams(train_tweets.tweetsText,w_ngram_min,w_ngram_max,c_ngram_min,c_ngram_max)

    # We didnt limit the maximum vocubulary length
    # and only removed words which occured less than 2 times in all the tweets
    vectoriser = TfidfVectorizer(analyzer=identity,min_df=2)
    train_vectors = vectoriser.fit_transform(train_features)

    with open('SVM 4 ngrams Results/vectoriser.pickle', 'wb') as handle:
        pickle.dump(vectoriser, handle, protocol=pickle.HIGHEST_PROTOCOL)


    test_features = convert_tweets_to_ngrams(test_tweets.tweetsText,w_ngram_min,w_ngram_max,c_ngram_min,c_ngram_max)
    test_vectors = vectoriser.transform(test_features)


    m = LinearSVC(C=0.1)
    m.fit(train_vectors,train_tweets.tweetsLabel)

    with open('SVM 4 ngrams Results/modelSVM.pickle', 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)

    predicated_labels= m.predict(test_vectors)

    official_evaluator(test_tweets.tweetsLabel,predicated_labels)
    evaluate_model("SVM",test_tweets.tweetsLabel,predicated_labels)





trainTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
tweets_with_location = True

train_tweets = get_train_data(trainTextDir, trainLabelDir, tweets_with_location)
test_tweets = get_test_data(testTextDir, testLabelDir,tweets_with_location)

run(train_tweets,test_tweets)