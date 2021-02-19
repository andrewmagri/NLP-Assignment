import pickle

from preprocessing import *
from WordEmbeddings import *

tweets = ""
labels = ""
tweet_list = []
option = 0


class Tweet:
    def __init__(self, text, label):
        self.text = text
        self.label = label


option = input("Enter option: ")
option = int(option)

if option == 1:
    # Obtaining tweet text
    with open(".\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text", "r",
              encoding="utf8") as t:
        tweets = t.read()
        tweets = tweets.split("\n")

    # Obtaining tweet label
    with open(".\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels", "r",
              encoding="utf8") as l:
        labels = l.read()
        labels = labels.split("\n")

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

        tweet_list.append(Tweet(newText, labels[i]))

    with open('Tweets.pickle', 'wb') as handle:
        pickle.dump(tweet_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif option == 2:
    with open('Tweets.pickle', 'rb') as handle:
        tweets = pickle.load(handle)

    with open(".\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text", "r",
              encoding="utf8") as t:
        tweetsOrig = t.read()
        tweetsOrig = tweetsOrig.split("\n")

    tweetText = []
    # Removing empty tweets due to removing locations
    for i, tweet in enumerate(tweets):
        if len(tweet.text) == 0:
            tweets.remove(tweet)
            continue
        tweetText.append(' '.join(tweet.text))


    matrix = word_embeddings(tweetText)
    print("ds")


