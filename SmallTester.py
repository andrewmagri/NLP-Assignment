from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from WordEmbeddings import *
from Scorer import *
from DataRetrival import *


def svm(tfidf_matrix, labels):
    clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
    clf.fit(tfidf_matrix, labels)
    return clf


def run(train_tweets, test_tweets):
    tfidf_featuriser = extract_tfidf_featuriser(train_tweets.tweetsText)
    train_tfidif_matrix = tfidf_featuriser.transform(train_tweets.tweetsText)
    test_tfidif_matrix = tfidf_featuriser.transform(test_tweets.tweetsText)
    clf = svm(train_tfidif_matrix, train_tweets.tweetsLabel)
    predictions = clf.predict(test_tfidif_matrix)
    save_model("Tester", tfidf_featuriser, clf)
    evaluate_model(clf, test_tweets.tweetsText, test_tweets.tweetsLabel, predictions)


trainTextDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
trainTweets = get_train_data(trainTextDir, trainLabelDir)
testTweets = get_test_data(testTextDir, testLabelDir)

trainTweets.tweetsText = trainTweets.tweetsText[:1000]
trainTweets.tweetsLabel = trainTweets.tweetsLabel[:1000]

testTweets.tweetsText = testTweets.tweetsText
testTweets.tweetsLabel = testTweets.tweetsLabel

run(trainTweets, testTweets)
