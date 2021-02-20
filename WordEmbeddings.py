from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def word_embeddings(tweets):
    tfidf_featuriser = TfidfVectorizer()
    tfidf_featuriser.fit(tweets)
    tfidf_matrix = tfidf_featuriser.transform(tweets)

    return tfidf_matrix