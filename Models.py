from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def naive_bayes_classifier(tfidf_matrix, labels):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(tfidf_matrix, labels)
    # to test the classifier
    # predictions = nb_classifier.predict(test_tfidf_docterm_matrix)




