tweets = ""
labels = ""
tweet_dict = []


class Tweet:
    def __init__(self, text, label):
        self.text = text
        self.label = label


with open(".\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels", "r") as t:
    tweets = t.read()

with open(".\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels", "r") as l:
    labels = l.read()

for i in range(0, len(tweets)):
    tweet_dict.append(Tweet(tweets[i], labels[i]))
