from DataRetrival import get_train_data, get_test_data
from Scorer import *
import random

def run(train_tweets, test_tweets):
    predicated_labels = []
    for test_tweet in test_tweets.tweetsText:
        predicated_labels.append(str(random.randint(0,19)))

    official_evaluator(test_tweets.tweetsLabel, predicated_labels)
    evaluate_model("SVM", test_tweets.tweetsLabel, predicated_labels)


trainTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
tweets_with_location = True

train_tweets = get_train_data(trainTextDir, trainLabelDir, tweets_with_location)
test_tweets = get_test_data(testTextDir, testLabelDir,tweets_with_location)

run(train_tweets,test_tweets)