from keras_preprocessing import sequence
from tensorflow.python.keras.utils.np_utils import to_categorical

vocab = {"<unk>": 0}
def preprocessing(tweets):
    tokenized = []
    labels = []
     # Start with the a unknown char in the vocad.

    for i, tweet in enumerate(tweets.tweetsText):
        part = []
        labels += [tweets.tweetsLabel[i]]
        tweetList = tweet.split()
        for word in tweetList:
            if word not in vocab:
                # add the unseen char to the vocab with a unique id.
                vocab[word] = len(vocab)
            part += [vocab[word]]
        tokenized += [part]

    max_chars = len(vocab)
    class_count = len(set(labels))

    # Padding :)
    tokenized = sequence.pad_sequences(tokenized, maxlen=10)

    # One hot encode the labels!
    labels = to_categorical(labels)

    vocab2 = vocab
    return (tokenized, labels)

def process_test_data( tweets):
    tokenized = []

    for text in tweets.tweetsText:
        part = []
        tweetList = text.split()
        for word in tweetList:
            # if c is in the vocab get the value for it otherwise use the unknown char(0)
            part += [vocab.get(word, 0)]
        tokenized += [part]

    tokenized = sequence.pad_sequences(tokenized, maxlen=10)

    return tokenized