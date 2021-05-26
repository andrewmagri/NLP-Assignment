from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_featuriser(tweets):
    # Fitting the vectoriser to the training data
    tfidf_featuriser = TfidfVectorizer()
    tfidf_featuriser.fit(tweets)

    return tfidf_featuriser