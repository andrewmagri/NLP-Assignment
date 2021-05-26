Data-Driven NLP Final Projects
Project 4: Multilingual EMOJI prediction

Authors: Andrew Joseph Magri, Daniel Attard

Implemented using Python 3.7, nltk 3.5, tensorflow 2.3 and scikit-learn 0.23.2
-------------------------------------------

The project is divided into a number of directories

The data folder contains the mappings, testing and training data.
The training data was obtained by crawling Twitter as mentioned on https://competitions.codalab.org/competitions/17344#learn_the_details-data
The testing data and mapping were also obtained from this link.

-------------------------------------------

The helpers directory stores any helper scripts such as DataRetrieval, Preprocessing, Scorer and WordEmbeddings which are self explanatory.

-------------------------------------------

All the implemented classifiers can be found in the Models folder including. NaiveBayes, Neural Network, SVM, Random Classifier and Random Forest models.

In order to run each model, the 'location' and 'language' variable are to be set as required.
The 'location' variable can be either True or False and defines whether location information is included in the tweet data (True) or not (False).
The 'language' variable can be either 'english' or 'spanish' to choose the required language.
The python file can be run normally and will output the macro F1, precision, recall scores along with the accuracy and a confusion matrix.
The False positive, false negative, true positive and true negative values are also outputted.


The preprocessed data for each language, with and without location information is stored in a pickle file in Models/TweetData.
The pretrained GloVe embeddings can be obtained from https://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip and were not included in this submission due to space limitations.
The file needs to be inserted into the Models/GloVe directory.
The Frequencies script can be run to obtain a breakdown of the percentage of the total labels.

-------------------------------------------