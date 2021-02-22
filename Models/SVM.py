from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from WordEmbeddings import *
from Scorer import *
from DataRetrival import *
from ModelPreprocessing import *


def svm(tfidf_matrix, labels):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(tfidf_matrix, labels)
    return clf


def run(trainTweets, testTweets):
    #tfidf_featuriser = extract_tfidf_featuriser(trainTweets.tweetsText)
    #train_tfidif_matrix = tfidf_featuriser.transform(trainTweets.tweetsText)
    #test_tfidif_matrix = tfidf_featuriser.transform(testTweets.tweetsText)
    y_test = testTweets.tweetsLabel

    (X_train, y_train) = preprocessing(trainTweets)
    X_test = process_test_data(testTweets)

    y_train =trainTweets.tweetsLabel

    clf = svm(X_train, y_train)
    predictions = clf.predict(X_test)
    evaluate_model("SVM",y_test, predictions)


trainTextDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
trainTweets = get_train_data(trainTextDir, trainLabelDir)
testTweets = get_test_data(testTextDir, testLabelDir)

run(trainTweets,testTweets)

