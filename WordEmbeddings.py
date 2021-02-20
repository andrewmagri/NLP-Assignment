from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_features(tweets):
    tfidf_featuriser = TfidfVectorizer()
    tfidf_featuriser.fit(tweets)
    tfidf_matrix = tfidf_featuriser.transform(tweets)

    return tfidf_matrix


def test_word2vec(text):
    model = Word2Vec(
        text,
        size=100,
        window=10,
        min_count=2,
        workers=10
    )
    model.train(text, total_examples=len(text), epochs=10)
    model.save('word2vec.model')
