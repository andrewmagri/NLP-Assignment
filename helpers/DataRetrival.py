from helpers.Preprocessing import *
import pickle

class Directory:
    def __init__(self):
        self.train_text_dir = ""
        self.train_label_dir = ""
        self.test_text_dir = ""
        self.test_label_dir = ""

        self.train_filename = ""
        self.test_filename = ""

        self.main_dir = "TweetData\\"
        self.location = None

        self.initial_retrieval = False



def set_directories_and_filename(dirs, location, language):
    if language == "english":
        dirs.train_filename = "Train_English_"
        dirs.train_text_dir = "../data/train/us_train.text"
        dirs.train_label_dir = "../data/train/us_train.labels"

        dirs.test_filename = "Test_English_"
        dirs.test_text_dir = "../data/test/us_test.text"
        dirs.test_label_dir = "../data/test/us_test.labels"

    elif language == "spanish":
        dirs.train_filename = "Train_Spanish_"
        dirs.train_text_dir = "../data/train/es_train.text"
        dirs.train_label_dir = "../data/train/es_train.labels"

        dirs.test_filename = "Test_Spanish_"
        dirs.test_text_dir = "../data/test/es_test.text"
        dirs.test_label_dir = "../data/test/es_test.labels"


    else:
        raise Exception("Incorrect Language Input")

    if location == True:
        dirs.train_filename += "with_location"
        dirs.test_filename += "with_location"
        dirs.location = True

    elif location == False:
        dirs.train_filename += "without_location"
        dirs.test_filename += "without_location"
        dirs.location = False

    else:
        raise Exception("Incorrect Location Input")

    return  dirs


# Checking if the preprocessed tweets already exist
def check_if_created(filename):
    try:
        file = open(filename + ".pickle")
        # file = open("..\\" +filename + ".pickle")
        file.close()
        return True
    except IOError:
        print("File not found")
        return False

# Creating the training or testing data by reading the tweet text and label and preprocessing the text
def get_data(dir_text,dir_label,location):
    # Obtaining tweet text
    with open(dir_text, "r",
              encoding="utf8") as t:
        tweets = t.read()
        tweets = tweets.split("\n")

    # Obtaining tweet label
    with open(dir_label, "r",
              encoding="utf8") as l:
        labels = l.read()
        labels = labels.split("\n")

    # Preprocessing the tweet text
    tweets_object = preprocess(tweets, labels,location)

    return tweets_object


# Obtaining the train data firstly by checking if it already exists else creating it.
def get_train_data(dirs):
    if check_if_created(dirs.main_dir + dirs.train_filename):
        dirs.initial_retrieval = False
        with open(dirs.main_dir + dirs.train_filename + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        dirs.initial_retrieval = True
        return get_data(dirs.train_text_dir, dirs.train_label_dir, dirs.location)


# Obtaining the test data firstly by checking if it already exists else creating it.
def get_test_data(dirs):
    if check_if_created(dirs.main_dir + dirs.test_filename):
        dirs.initial_retrieval = False
        with open(dirs.main_dir + dirs.test_filename + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        dirs.initial_retrieval = True
        return get_data(dirs.test_text_dir, dirs.test_label_dir, dirs.location)


def get_tweet_data(location, language):
    dirs = Directory()

    dirs = set_directories_and_filename(dirs, location, language)

    train_data = get_train_data(dirs)
    test_data = get_test_data(dirs)

    # If the data is obtained from the raw files, check for duplicates
    if dirs.initial_retrieval:
        # Checking for duplicate tweets in the train and test sets and removing them from the test set
        counter = 0
        for idx, testTweet in enumerate(test_data.tweetsText):
            for trainTweet in train_data.tweetsText:
                if trainTweet == testTweet:
                    counter += 1
                    test_data.tweetsText.pop(idx)
                    test_data.tweetsLabel.pop(idx)
                    break

        print("Number of tweets found both in train and test set", str(counter))
        print("Removed duplicate tweets from train set")


        # Saving the train and test tweets
        with open(dirs.main_dir + dirs.train_filename+'.pickle', 'wb') as handle:
            pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dirs.main_dir + dirs.test_filename+'.pickle', 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_data,test_data

