from helpers.DataRetrival import *

# Outputs the frequency of each label in the obtained data
train_tweets, test_tweets = get_tweet_data(True, "english")

label_dict = {}

for tweetLabel in train_tweets.tweetsLabel:
    if tweetLabel in label_dict:
        label_dict[tweetLabel] += 1
    else:
        label_dict[tweetLabel] = 0

total_no_of_labels = 387792
for label in label_dict:
    print("Label " + label + " : {:.3%} %".format(label_dict[label]/total_no_of_labels))
