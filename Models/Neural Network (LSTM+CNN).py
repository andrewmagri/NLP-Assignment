from keras.layers import LSTM, Dense, Dropout, Embedding, Activation, Conv1D, MaxPooling1D
from keras.models import Sequential
from helpers.DataRetrival import get_tweet_data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from helpers.Scorer import *


# loading the pretrained word embeddings from file.
def get_embeddings_dict():
    embeddings_dict = {}
    f = open('GloVe/glove.twitter.27B.100d.txt', encoding="utf8")
    for embedding in f:
        values = embedding.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = coefs
    f.close()
    return embeddings_dict


# generating the embedding matrix
def create_embeddings(maxwords, word_index, embedding_size):
    embeddings_dict = get_embeddings_dict()
    num_words = min(maxwords, len(word_index))
    embedding_matrix = np.zeros((num_words, embedding_size))
    # adding the vector representing each word to the matrix
    for word, i in word_index.items():
        if i >= maxwords:
            continue
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# method to tokenize the training data
def tokenize_train_data(tokenizer, tweet_text, tweet_labels, maxlen):
    tweet_labels = to_categorical(tweet_labels)

    tokenizer.fit_on_texts(tweet_text)
    tokenized = tokenizer.texts_to_sequences(tweet_text)
    tokenized = pad_sequences(tokenized, maxlen=maxlen)

    return tokenized, tweet_labels, tokenizer


# method to tokenize the test data
def tokenize_test_data(tokenizer, tweet_text):
    tokenized = tokenizer.texts_to_sequences(tweet_text)
    tokenized = pad_sequences(tokenized, maxlen=maxlen)
    return tokenized


def lstm(maxlen, maxwords, class_count, embedding_size, filters, kernel_size, pool_size, lstm_size, embedding_matrix):
    # this lstm model is composed of the following layers: Embedding, Dropout, Convolutional, Max Pooling, LSTM, Dense,
    # and Activation
    model = Sequential()
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


def run(train_tweets, test_tweets,language):
    tokenizer = Tokenizer(num_words=maxwords)
    # tokenizing the data
    x_train, y_train, tokenizer = tokenize_train_data(tokenizer, train_tweets.tweetsText, train_tweets.tweetsLabel, maxlen)
    # getting pre-trained embeddings in matrix format
    embedding_matrix = create_embeddings(maxwords, tokenizer.word_index, embedding_size)
    # creating the model
    model = lstm(maxlen, maxwords, class_count, embedding_size, filters, kernel_size, pool_size, lstm_size, embedding_matrix)
    # fitting the model with the training data
    model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs)

    test_data = tokenize_test_data(tokenizer, test_tweets.tweetsText)

    # predicting the labels of the test data
    predicated_labels = model.predict(test_data)

    evaluate_model("LSTM", test_tweets.tweetsLabel, predicated_labels.argmax(axis=1).astype(str), language)


maxlen = 20
maxwords = 20000
epochs = 20
embedding_size = 100
lstm_size = 64
kernel_size = 5
filters = 64
pool_size = 4
class_count = 20
batch_size = 32

# Setting if the location is required and which language of tweets to obtain
location = False
language = "spanish"

if language == "english":
    class_count = 20
elif language == "spanish":
    class_count = 19

train_tweets, test_tweets = get_tweet_data(location,language)

# Running the SVM models on the train and test tweets
run(train_tweets, test_tweets, language)
