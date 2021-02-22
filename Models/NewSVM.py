import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from DataRetrival import get_train_data, get_test_data
from nltk import ngrams
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
def svm():
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(tfidf_matrix, labels)
    return clf

#https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
def flatten_list(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def get_ngram(tweets_text, min, max):
    output = []
    for tweetText in tweets_text:
        for n in range (min ,max+1):
            ngramOutput = list(ngrams(tweetText.split(),n))
            if len(ngramOutput) !=0:
                output.append(ngramOutput)
    return flatten_list(output)

def get_word_ngrams(tweets_text, n):
    return get_ngram(tweets_text,1,4)



def run(train_tweets, test_tweets):
    n_ngram = 4
    features = get_word_ngrams(train_tweets.tweetsText,n_ngram)

    # We didnt limit the maximum vocubulary length
    # and only removed words which occured less than 2 times in all the tweets
    vectoriser = TfidfVectorizer(min_df=2)
    vectors = vectoriser.fit_transform(features)

    with open('vectoriser.pickle', 'wb') as handle:
        pickle.dump(vectoriser, handle, protocol=pickle.HIGHEST_PROTOCOL)



trainTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
train_tweets = get_train_data(trainTextDir, trainLabelDir)
#test_tweets = get_test_data(testTextDir, testLabelDir)

train_tweets.tweetsText = train_tweets.tweetsText[:10]
train_tweets.tweetsLabel = train_tweets.tweetsLabel[:10]

run(train_tweets,None)