import pickle

import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,plot_confusion_matrix
import matplotlib.pyplot as plt


def get_class_names():
    output_classes = []
    with open("..\\Semeval2018-Task2-EmojiPrediction\\mapping\\us_mapping.txt", "r",encoding="utf8") as l:
        classes = l.read()
        classes = classes.split("\n")
        for className in classes:
            classText = className.split("\t")
            output_classes.append(''.join(classText[0]))

    return output_classes


def evaluate_model(modelName, true_labels, predicted_labels):
    #print(confusion_matrix(true_labels, predicted_labels,labels=get_class_names()))
    print("Precision: " + str(precision_score(true_labels, predicted_labels,average="micro")))
    print("Recall: " + str(recall_score(true_labels, predicted_labels,average="micro")))
    print("F1 Score: " + str(f1_score(true_labels, predicted_labels,average="micro")))
    accuracy = np.sum(predicted_labels == true_labels)/len(true_labels)
    print('Accuracy: {:.3%}'.format(accuracy))
    confusion_matrixActual = confusion_matrix(true_labels, predicted_labels)
    FP = confusion_matrixActual.sum(axis=0) - np.diag(confusion_matrixActual)
    FN = confusion_matrixActual.sum(axis=0) - np.diag(confusion_matrixActual)
    TP = np.diag(confusion_matrixActual)
    print("FP", confusion_matrixActual.sum(axis=0) - np.diag(confusion_matrixActual))
    print("FN",confusion_matrixActual.sum(axis=1) - np.diag(confusion_matrixActual))
    print("TP",np.diag(confusion_matrixActual))
    #print("TN",confusion_matrixActual.sum() - (FP + FN + TP))



    labels = get_class_names()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrixActual)
    plt.title('Confusion matrix of the classifier' + modelName)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.locator_params(axis="y",nbins = 20)
    plt.locator_params(axis="x",nbins = 20)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def save_model(model_name, tfidf_featuriser, model):
    pickle.dump(model, open("ModelsOutput/" + model_name + "Model", 'wb'))
    pickle.dump(tfidf_featuriser, open("ModelsOutput/" + model_name + "tfidf_featuriser", 'wb'))