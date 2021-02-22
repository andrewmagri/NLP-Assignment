# https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html
import datetime

from keras.layers import LSTM, Dense, Dropout, Embedding, Activation, Bidirectional
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorboard.plugins.hparams import api as hp
from DataRetrival import *

import tensorflow as tf

from WordEmbeddings import extract_tfidf_featuriser

N = 1000
Tp = 800
epochs = 1000
noOfNeuronsInOutputLayer = 7
dateAndTimeNow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Obtaining the csv file having all outputs on 1 coloumn
# X49LagAccData,y49lagAccData = returnXYData(49, 5)

# X49LagAccData.to_pickle("X49LagAccData")
# y49lagAccData.to_pickle("y49lagAccData")

trainTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "..\\Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
train_tweets = get_train_data(trainTextDir, trainLabelDir)
test_tweets = get_test_data(testTextDir, testLabelDir)

X_train = train_tweets.tweetsText[:100]
y_train = train_tweets.tweetsLabel[:100]


X_test = test_tweets.tweetsText
y_test = test_tweets.tweetsLabel


tfidf_featuriser = extract_tfidf_featuriser(train_tweets.tweetsText[:100])
X_train = tfidf_featuriser.transform(train_tweets.tweetsText[:100])
X_test = tfidf_featuriser.transform(test_tweets.tweetsText[:100])

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([7]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
HP_ACT_FUNC = hp.HParam('activation_function', hp.Discrete(["softmax"]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(["rmsprop"]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning/' + dateAndTimeNow).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_BATCH_SIZE, HP_ACT_FUNC],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )


def lstm(hparams, input_dimension, output_dimension=300, max_length):
    step = X_train.shape[1]
    model = Sequential()
    model.add(Embedding(input_dimension,
                        output_dimension,
                        input_length=max_length))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(input_shape=(step, 1), units=10, activation="softmax")))
    model.add(Bidirectional(LSTM(input_shape=(step, 1), units=10, activation="softmax")))
    model.add(Dense(20, activation="softmax"))
    model.add(Activation('softmax'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['acc'])
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=hparams[HP_BATCH_SIZE])  # Run with 1 epoch to speed things up for demo purposes
    test_predictions = model.predict(X_test)
    _, accuracy = model.evaluate(X_test, y_test)

    return accuracy


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model2(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
        for act_func in HP_ACT_FUNC.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_BATCH_SIZE: batch_size,
                    HP_ACT_FUNC: act_func,
                    HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + dateAndTimeNow + run_name, hparams)
                session_num += 1
