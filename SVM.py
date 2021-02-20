from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm(tfidf_matrix, labels):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(tfidf_matrix, labels)
    return clf


def run():
    pass
