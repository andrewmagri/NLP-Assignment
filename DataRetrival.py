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


def get_test_data(dirTestText, dirTestLabel):
    filename = "TestTweets"
    if check_if_created(filename):
        with open(filename + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return get_data(dirTestText, dirTestLabel, filename)


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