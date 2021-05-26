import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,plot_confusion_matrix
import matplotlib.pyplot as plt



def get_class_names(language):
    output_classes = []
    # with open("..\\Semeval2018-Task2-EmojiPrediction\\mapping\\us_mapping.txt", "r",encoding="utf8") as l:
    if language == "english":
        with open("..\\data\\mapping\\us_mapping.txt", "r", encoding="utf8") as l:
            classes = l.read()
            classes = classes.split("\n")
            for className in classes:
                classText = className.split("\t")
                output_classes.append(''.join(classText[0]))

    elif language == "spanish":
        with open("..\\data\\mapping\\es_mapping.txt", "r", encoding="utf8") as l:
            classes = l.read()
            classes = classes.split("\n")
            for className in classes:
                classText = className.split("\t")
                output_classes.append(''.join(classText[0]))

    else:
        raise Exception("Incorrect Language Input")

    return output_classes

# Adapated from #https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/blob/master/tools/evaluation%20script/scorer_semeval18.py

def evaluate_model(modelName, true_labels, predicted_labels,language):
    # Obtaing the precision, recall and F1 Score with a 'macro' average while also showing the accuracy
    #print(confusion_matrix(true_labels, predicted_labels,labels=get_class_names()))
    print("Macro Precision: " + str(precision_score(true_labels, predicted_labels,average="macro")))
    print("Macro Recall: " + str(recall_score(true_labels, predicted_labels,average="macro")))
    print("Macro F1 Score: " + str(f1_score(true_labels, predicted_labels,average="macro")))
    accuracy = np.sum(predicted_labels == true_labels)/len(true_labels)
    print('Accuracy: {:.3%}'.format(accuracy))

    confusion_matrixActual = confusion_matrix(true_labels, predicted_labels)

    # Displaying the respective FP, FN, TP
    FP = confusion_matrixActual.sum(axis=0) - np.diag(confusion_matrixActual)
    FN = confusion_matrixActual.sum(axis=1) - np.diag(confusion_matrixActual)
    TP = np.diag(confusion_matrixActual)

    print("False Positives ", confusion_matrixActual.sum(axis=0) - np.diag(confusion_matrixActual))
    print("False Negatives ",confusion_matrixActual.sum(axis=1) - np.diag(confusion_matrixActual))
    print("True Positives ",np.diag(confusion_matrixActual))
    print("True Negatives ",confusion_matrixActual.sum() - (FP + FN + TP))



    # Plotting the confusion matrix
    labels = get_class_names(language)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrixActual)
    plt.title('Confusion matrix for ' + modelName)
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