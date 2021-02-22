import pickle

import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,plot_confusion_matrix
import matplotlib.pyplot as plt

#https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/blob/master/tools/evaluation%20script/scorer_semeval18.py
def f1(precision,recall):
    return (2.0*precision*recall)/(precision+recall)


def main(path_goldstandard, path_outputfile):

    truth_dict={}
    output_dict_correct={}
    output_dict_attempted={}
    truth_file_lines=open(path_goldstandard,encoding='utf8').readlines()
    submission_file_lines=open(path_outputfile,encoding='utf8').readlines()
    if len(submission_file_lines)!=len(truth_file_lines): print("Inconsistent number of inputs")
    for i in range(len(submission_file_lines)):
        line=submission_file_lines[i]
        emoji_code_gold=truth_file_lines[i].replace("\n","")
        if emoji_code_gold not in truth_dict: truth_dict[emoji_code_gold]=1
        else: truth_dict[emoji_code_gold]+=1
        emoji_code_output=submission_file_lines[i].replace("\n","")
        if emoji_code_output==emoji_code_gold:
            if emoji_code_output not in output_dict_correct: output_dict_correct[emoji_code_gold]=1
            else: output_dict_correct[emoji_code_output]+=1
        if emoji_code_output not in output_dict_attempted: output_dict_attempted[emoji_code_output]=1
        else: output_dict_attempted[emoji_code_output]+=1
    precision_total=0
    recall_total=0
    num_emojis=len(truth_dict)
    attempted_total=0
    correct_total=0
    gold_occurrences_total=0
    f1_total=0
    for emoji_code in truth_dict:
        gold_occurrences=truth_dict[emoji_code]
        if emoji_code in output_dict_attempted: attempted=output_dict_attempted[emoji_code]
        else: attempted=0
        if emoji_code in output_dict_correct: correct=output_dict_correct[emoji_code]
        else: correct=0
        if attempted!=0:
            precision=(correct*1.0)/attempted
            recall=(correct*1.0)/gold_occurrences
            if precision!=0.0 or recall!=0.0: f1_total+=f1(precision,recall)
        attempted_total+=attempted
        correct_total+=correct
        gold_occurrences_total+=gold_occurrences
    macrof1=f1_total/(num_emojis*1.0)
    precision_total_micro=(correct_total*1.0)/attempted_total
    recall_total_micro=(correct_total*1.0)/gold_occurrences_total
    if precision_total_micro!=0.0 or recall_total_micro!=0.0: microf1=f1(precision_total_micro,recall_total_micro)
    else: microf1=0.0
    print ("Macro F-Score (official): "+str(round(macrof1*100,3)))
    print ("-----")
    print ("Micro F-Score: "+str(round(microf1*100,3)))
    print ("Precision: "+str(round(precision_total_micro*100,3)))
    print ("Recall: "+str(round(recall_total_micro*100,3)))



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
    print("Precision: " + str(precision_score(true_labels, predicted_labels,average="macro")))
    print("Recall: " + str(recall_score(true_labels, predicted_labels,average="macro")))
    print("F1 Score: " + str(f1_score(true_labels, predicted_labels,average="macro")))
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