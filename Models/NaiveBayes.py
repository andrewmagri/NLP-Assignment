from sklearn.naive_bayes import MultinomialNB
from WordEmbeddings import *
from DataRetrival import *
from Scorer import *


def naive_bayes_classifier(tfidf_matrix, labels):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(tfidf_matrix, labels)
    return nb_classifier


def run(trainTweets, testTweets):
    tfidf_featuriser = extract_tfidf_featuriser(trainTweets.tweetsText)
    train_tfidif_matrix = tfidf_featuriser.transform(trainTweets.tweetsText)
    test_tfidif_matrix = tfidf_featuriser.transform(testTweets.tweetsText)
    clf = naive_bayes_classifier(train_tfidif_matrix, trainTweets.tweetsLabel)
    predictions = clf.predict(test_tfidif_matrix)
    evaluate_model(clf, testTweets.tweetsText, testTweets.tweetsLabel,predictions)


trainTextDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"

trainTweets = get_train_data(trainTextDir, trainLabelDir)
testTweets = get_test_data(testTextDir, testLabelDir)
run(trainTweets,testTweets)

