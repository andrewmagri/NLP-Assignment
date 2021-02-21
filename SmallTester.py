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


def run(trainTweets,testTweets):
    tfidf_featuriser = extract_tfidf_featuriser(trainTweets.tweetsText)
    train_tfidif_matrix = tfidf_featuriser.transform(trainTweets.tweetsText)
    test_tfidif_matrix = tfidf_featuriser.transform(testTweets.tweetsText)
    clf = svm(train_tfidif_matrix, trainTweets.tweetsLabel)
    predictions = clf.predict(test_tfidif_matrix)
    evaluate_model(clf,testTweets.tweetsText,testTweets.tweetsLabel,predictions)


trainTextDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
trainTweets = get_train_data(trainTextDir, trainLabelDir)
testTweets = get_test_data(testTextDir, testLabelDir)


trainTweets.tweetsText = trainTweets.tweetsText[:1000]
trainTweets.tweetsLabel = trainTweets.tweetsLabel[:1000]

run(trainTweets,testTweets)
