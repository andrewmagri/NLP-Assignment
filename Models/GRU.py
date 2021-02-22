# https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html
import datetime
import time
import sklearn
from keras.layers import GRU, Dense, Embedding, Dropout
from keras.models import Sequential
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorboard.plugins.hparams import api as hp
from DataRetrival import *
from Scorer import *
from ModelPreprocessing import *

import tensorflow as tf
import numpy as np
import matplotlib as plt
from WordEmbeddings import extract_tfidf_featuriser




N = 1000
Tp = 800
epochs = 500
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

X_train = train_tweets.tweetsText
y_train = train_tweets.tweetsLabel

X_test = test_tweets.tweetsText
y_test = test_tweets.tweetsLabel


(X_train,y_train) = preprocessing(train_tweets)
X_test = process_test_data(test_tweets)
y_test = to_categorical(y_test)


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([15]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32,64]))
HP_ACT_FUNC = hp.HParam('activation_function', hp.Discrete(["softmax"]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(["rmsprop"]))

METRIC_ACCURACY = 'accuracy'



with tf.summary.create_file_writer('logs/hparam_tuning/' + dateAndTimeNow).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_BATCH_SIZE, HP_ACT_FUNC],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )


def train_test_model2(hparams):

    start_time = time.time()
    input = Input(shape=(X_train.shape[1:]))
    output = tf.expand_dims(input,axis = -1)
    output = GRU(units=hparams[HP_NUM_UNITS],activation="relu",return_sequences=True)(output)
    output = GRU(units=hparams[HP_NUM_UNITS], activation="relu",)(output)

    #dropout_out = Dropout(0.8)(output)

    # Adding some further layers (replace or remove with your architecture):
    out = Dense(units=20, activation="relu")(output)



    # Building model:
    model = Model(inputs=input, outputs=out)
    model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=['acc'])
    print(model.summary())
    plot_model(model, to_file='model_plot' + dateAndTimeNow + '.png', show_shapes=True, show_layer_names=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=hparams[HP_BATCH_SIZE],verbose=1,callbacks=[
        tf.keras.callbacks.TensorBoard(
            "logs/fit/" + dateAndTimeNow)])  # Run with 1 epoch to speed things up for demo purposes
    test_predictions = model.predict(X_test)

    y_test2 = np.argmax(y_test, axis=1)
    test_predictions = np.round(test_predictions)
    sums = np.sum(test_predictions, axis=1)

    test_predictions2 = np.argmax(test_predictions, axis=1)
    #print(sklearn.metrics.multilabel_confusion_matrix(y_test2, test_predictions2))
    _, accuracy = model.evaluate(X_test, y_test)
    evaluate_model("GRU"+str(hparams[HP_NUM_UNITS]), y_test2, test_predictions2)
    print("Total Time" + str(time.time()-start_time))
    
    """
    # Best 0.96
    #step = X_train.shape[1]
    model = Sequential()
    #model.add(Embedding(10000,300,input_length=1))
    model.add(GRU(input_shape=(100), units=10, activation="softmax"))
    model.add(Dense(20, activation="softmax"))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['acc'])
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=hparams[HP_BATCH_SIZE])  # Run with 1 epoch to speed things up for demo purposes
    test_predictions = model.predict(X_test)
    

    """

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
