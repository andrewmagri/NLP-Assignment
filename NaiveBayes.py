from sklearn.naive_bayes import MultinomialNB
from WordEmbeddings import *
from Preprocessing import *
from Tweets import *

import pickle


def check_if_created(filename):
    try:
        file = open(filename + ".pickle")
        file.close()
        return True
    except IOError:
        print("File not found")
        return False


def get_data(dirText, dirLabel,output_file_name):
    # Obtaining tweet text
    with open(dirText, "r",
              encoding="utf8") as t:
        tweets = t.read()
        tweets = tweets.split("\n")

    # Obtaining tweet label
    with open(dirLabel, "r",
              encoding="utf8") as l:
        labels = l.read()
        labels = labels.split("\n")

    tweets_object = preprocess(tweets, labels)

    with open(output_file_name+'.pickle', 'wb') as handle:
        pickle.dump(tweets_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tweets_object


def get_train_data(dirTrainText, dirTrainLabel):
    filename = "TrainTweets"
    if check_if_created(filename):
        with open(filename + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return get_data(dirTrainText, dirTrainLabel,filename)


def preprocess(tweets,labels):
    tweets_object = Tweets()
    for i in range(0, len(tweets)):
        tweets[i] = " ".join(tweets[i].split())
        tweets[i] = tokenize(tweets[i])

        newText = []
        for word in tweets[i]:
            # Checking for @ Location and eliminating any words that follow
            if word == "@":
                break

            word = lemmatise(word)
            word = remove_stopwords(word)
            word = remove_url(word)
            word = remove_puncuation(word)

            if word != "" and word is not None:
                newText.append(word)

        if len(newText) == 0:
            continue

        tweets_object.tweetsText.append(' '.join(newText))
        tweets_object.tweetsLabel.append(labels[i])
    return tweets_object


def naive_bayes_classifier(tfidf_matrix, labels):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(tfidf_matrix, labels)
    return nb_classifier
    # to test the classifier
    # predictions = nb_classifier.predict(test_tfidf_docterm_matrix)


def run():
    print()


trainTextDir ="Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
# testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
# testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
trainTweets = get_train_data(trainTextDir, trainLabelDir)
# testTweets = get_test_data(testTextDir, testLabelDir)
tfidf_matrix = extract_tfidf_features(trainTweets.tweetsText)
clf = run(tfidf_matrix, trainTweets.labels)
