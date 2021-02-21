import pickle

from DataRetrival import get_train_data, get_test_data
from Scorer import evaluate_model
from WordEmbeddings import extract_tfidf_featuriser

model = pickle.load(open("ModelsOutput/TesterModel", 'rb'))
featuriser = pickle.load(open("ModelsOutput/Testertfidf_featuriser", 'rb'))
trainTextDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
train_tweets = get_train_data(trainTextDir, trainLabelDir)
test_tweets = get_test_data(testTextDir, testLabelDir)

test_tfidif_matrix = featuriser.transform(test_tweets.tweetsText)

predictions = model.predict(test_tfidif_matrix)
evaluate_model(model, test_tweets.tweetsText, test_tweets.tweetsLabel, predictions)
