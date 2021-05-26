from sklearn.ensemble import RandomForestClassifier
from helpers.WordEmbeddings import *
from helpers.Scorer import *
from helpers.DataRetrival import *


def random_forest(tfidf_matrix, labels):
    # Initializing a random forest classifier and fitting the tfidf matrix
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(tfidf_matrix, labels)
    return rf_classifier


def run_random_forest(trainTweets, testTweets, language):
    # Generating the tf idf matrix and transforming the train and test data
    tfidf_featuriser = extract_tfidf_featuriser(trainTweets.tweetsText)
    train_tfidif_matrix = tfidf_featuriser.transform(trainTweets.tweetsText)
    test_tfidif_matrix = tfidf_featuriser.transform(testTweets.tweetsText)

    clf = random_forest(train_tfidif_matrix, trainTweets.tweetsLabel)
    predictions = clf.predict(test_tfidif_matrix)

    evaluate_model("Random Forest", testTweets.tweetsLabel, predictions, language)


# Setting if the location is required and which language of tweets to obtain
location = False
language = "english"

train_tweets, test_tweets = get_tweet_data(location, language)

# Running the SVM models on the train and test tweets
run_random_forest(train_tweets, test_tweets, language)
