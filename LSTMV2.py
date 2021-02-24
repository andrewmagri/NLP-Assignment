from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import LSTM, Dense, Dropout, Embedding, Activation, Bidirectional, Conv1D, MaxPooling1D
from keras.models import Sequential
from DataRetrival import get_train_data, get_test_data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk import ngrams
from sklearn.svm import LinearSVC
from Scorer import *


def get_embeddings_index():
    embeddings_index = {}
    f = open('GloVe/glove.twitter.27B.100d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def create_embeddings(maxwords, word_index, embedding_size):
    embeddings_index = get_embeddings_index()
    num_words = min(maxwords, len(word_index))
    embedding_matrix = np.zeros((num_words, embedding_size))
    for word, i in word_index.items():
        if i >= maxwords:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def tokenize_train_data(tokenizer, tweet_text, tweet_labels, maxlen):
    tweet_labels = to_categorical(tweet_labels)

    tokenizer.fit_on_texts(tweet_text)
    # word_index = tokenizer.word_index
    tokenized = tokenizer.texts_to_sequences(tweet_text)
    tokenized = pad_sequences(tokenized, maxlen=maxlen)

    # embedding_matrix = create_embeddings(maxwords, word_index, embedding_size)

    return tokenized, tweet_labels, tokenizer


def tokenize_test_data(tokenizer, tweet_text):
    tokenized = tokenizer.texts_to_sequences(tweet_text)
    tokenized = pad_sequences(tokenized, maxlen=maxlen)
    return tokenized



def lstm(maxlen, maxwords, class_count, embedding_size, filters, kernel_size, pool_size, lstm_size, embedding_matrix):
    model = Sequential()
    # model.add(Embedding(num_words,
    #                     embedding_size,
    #                     weights=[embedding_matrix],
    #                     input_length=maxlen,
    #                     trainable=False))
    model.add(Embedding(maxwords,
                        embedding_size,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_size))
    model.add(Dense(class_count))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

    # step = X_train.shape[1]
    # model = Sequential()
    # model.add(Embedding(input_dimension,
    #                     output_dimension,
    #                     input_length=max_length))
    # model.add(Dropout(0.25))
    # model.add(Bidirectional(LSTM(input_shape=(step, 1), units=10, activation="softmax")))
    # model.add(Bidirectional(LSTM(input_shape=(step, 1), units=10, activation="softmax")))
    # model.add(Dense(20, activation="softmax"))
    # model.add(Activation('softmax'))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['acc'])
    # model.summary()
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #
    # model.fit(X_train, y_train, epochs=epochs, batch_size=hparams[HP_BATCH_SIZE])  # Run with 1 epoch to speed things up for demo purposes
    # test_predictions = model.predict(X_test)
    # _, accuracy = model.evaluate(X_test, y_test)
    #
    # return accuracy


def run(train_tweets, test_tweets):
    tokenizer = Tokenizer(num_words=20000)

    x_train, y_train, tokenizer = tokenize_train_data(tokenizer, train_tweets.tweetsText, train_tweets.tweetsLabel, maxlen)

    embedding_matrix = create_embeddings(maxwords, tokenizer.word_index, embedding_size)

    model = lstm(maxlen, maxwords, class_count, embedding_size, filters, kernel_size, pool_size, lstm_size, embedding_matrix)

    model.fit(x_train, y_train,
                   batch_size=32,
                   epochs=20)

    test_data = tokenize_test_data(tokenizer, test_tweets.tweetsText)
    # predicated_labels = model.predict_on_batch([tweet for tweet in test_tweets.tweetsText])
    predicated_labels = model.predict(test_data)

    # official_evaluator(test_tweets.tweetsLabel, predicated_labels.argmax(axis=1))
    evaluate_model("LSTM", test_tweets.tweetsLabel, predicated_labels.argmax(axis=1).astype(str))


maxlen = 20
maxwords = 20000
epochs = 20
iteration_scoring = True
checkpoint_saving = True
max_non_improving_iterations = 5
embedding_size = 100
lstm_size = 64
kernel_size = 5
filters = 64
pool_size = 4
activation = 'sigmoid'
optimizer = 'adam'
class_count = 20

trainTextDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.text"
trainLabelDir = "Semeval2018-Task2-EmojiPrediction\\Data\\tweet_by_ID_04_2_2021__05_27_42.txt.labels"
testTextDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.text"
testLabelDir = "Semeval2018-Task2-EmojiPrediction\\test\\us_test.labels"
train_tweets = get_train_data(trainTextDir, trainLabelDir)
test_tweets = get_test_data(testTextDir, testLabelDir)

train_tweets.tweetsText = train_tweets.tweetsText
train_tweets.tweetsLabel = train_tweets.tweetsLabel

run(train_tweets,test_tweets)
