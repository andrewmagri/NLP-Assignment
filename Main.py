import pickle
import os

from sklearn.model_selection import train_test_split
from Preprocessing import *
from WordEmbeddings import *
from Models import *
from Tweets import *
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

def check_if_created(filename):
    try:
        file = open(filename + ".pickle")
        file.close()
        return True
    except IOError:
        print("File not found")
        return False


def get_train_data(dirTrainText, dirTrainLabel):
    filename = "TrainTweets"
    if check_if_created(filename):
        with open(filename + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return get_data(dirTrainText, dirTrainLabel,filename)


def get_test_data(dirTestText, dirTestLabel):
    filename = "TestTweets"
    if check_if_created(filename):
        with open(filename + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return get_data(dirTestText, dirTestLabel, filename)


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

def menu(option):
    if option == 1:
        trainTextDir ="Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
        trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"

        testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
        testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"

        trainTweets = get_train_data(trainTextDir, trainLabelDir)
        testTweets = get_test_data(testTextDir, testLabelDir)




        tfidf_matrix = extract_tfidf_features(trainTweets.tweetsText)
        model = svm(tfidf_matrix,trainTweets.tweetsLabel)

        predicated_labels = model.predict(testTweets.tweetsText)

        print(confusion_matrix(testTweets.tweetsLabel,predicated_labels))
        print("Precision: " + precision_score(testTweets.tweetsLabel,predicated_labels,average="macro"))
        print("Recall: " + recall_score(testTweets.tweetsLabel,predicated_labels,average="macro"))
        print("F1 Score: " + f1_score(testTweets.tweetsLabel,predicated_labels,average="macro"))
    else:
        print("L input hazin")





option = input("Enter option: ")
option = int(option)
menu(option)

