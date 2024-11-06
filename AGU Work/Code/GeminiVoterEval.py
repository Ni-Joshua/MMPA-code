import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
from functools import reduce
import google.generativeai as genai
import time

def evalfirst(Predictions_GT):
    predictionans = []
    print(len(Predictions_GT))
    for i in range(0, len(Predictions_GT)):
        locations = {
            "Similar Active Fire": np.inf,
            "Smaller Active Fire": np.inf,
            "Larger Active Fire": np.inf,
            "No New Active Fire": np.inf,
            "No Active Fire": np.inf,
            "New Active Fire": np.inf,
        }
        row = Predictions_GT.iloc[i]
        for key in locations.keys():
            try:
                locations[key] = row['Decision_LLM46'].index(key)
            except Exception:
                continue
        predictionans.append(min(locations, key=locations.get))
        # print(locations, min(locations, key=locations.get))
    Predictions_GT.insert(5,"Prediction Answer" ,predictionans)
    y_true = Predictions_GT['Ground_Truth']
    y_pred = Predictions_GT['Prediction Answer']
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    Predictions_GT.to_csv("GeminiVoterResultsandAns.csv", index=False)
    return accuracy, precision, recall, f1


def eval(Predictions_GT):
    # predictionans = []
    # print(len(Predictions_GT))
    # for i in range(0, len(Predictions_GT)):
    #     locations = {
    #         "Similar Active Fire": np.inf,
    #         "Smaller Active Fire": np.inf,
    #         "Larger Active Fire": np.inf,
    #         "No New Active Fire": np.inf,
    #         "No Active Fire": np.inf,
    #         "New Active Fire": np.inf,
    #     }
    #     row = Predictions_GT.iloc[i]
    #     for key in locations.keys():
    #         try:
    #             locations[key] = row['Decision_LLM46'].index(key)
    #         except Exception:
    #             continue
    #     predictionans.append(min(locations, key=locations.get))
    #     # print(locations, min(locations, key=locations.get))
    # Predictions_GT.insert(5,"Prediction Answer" ,predictionans)
    y_true = Predictions_GT['Ground_Truth']
    y_pred = Predictions_GT['Prediction Answer']
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    # Predictions_GT.to_csv("GeminiVoterResultsandAns.csv", index=False)
    return accuracy, precision, recall, f1

def plot_cmatrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["No Active Fire", "Smaller Active Fire",  "Similar Active Fire","Larger Active Fire",  "New Active Fire", "No New Active Fire",])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No",  "Smaller","Similar", "Larger", "New", "No New",])
    
    disp.plot()
    fig = disp.ax_.get_figure() 
    fig.set_figwidth(13)
    fig.set_figheight(10) 
    plt.suptitle(title)

# predans = pd.read_csv("GeminiVoterResultsandAns(3Groups)(LLMVoter).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "3Groups, LLM Voter \n" + str(eval(predans)))

# predans = pd.read_csv("GeminiVoterResultsandAns(OrigGroups)(LLMVoter).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "OrigGroups, LLM Voter \n"+ str(eval(predans)))

# predans = pd.read_csv("GeminiVoterResultsandAns(GeminiGroups)(LLMVoter).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "GeminiGroups, LLM Voter \n"+ str(eval(predans)))

# predans = pd.read_csv("GeminiVoterResultsandAns(3Groups).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "3Groups \n"+ str(eval(predans)))

# predans = pd.read_csv("GeminiVoterResultsandAns(OrigGroups).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "OrigGroups \n"+ str(eval(predans)))

# predans = pd.read_csv("GeminiVoterResultsandAns(GeminiGroups).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "GeminiGroups \n"+ str(eval(predans)))


# predans = pd.read_csv("GeminiVoterResultsandAns(5 Shots, ChoiceEx).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "5 Shots, Choice Ex, Orig Run \n"+ str(eval(predans)))




# print(eval(pd.read_csv("GeminiVoterResultsandAns(G4,LLMVoter,Ex,Shuffle).csv")))

# predans = pd.read_csv("GeminiVoterResultsandAns(NoVoterPatched).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "NoVoter Patched \n"+ str(eval(predans)))

# predans = pd.read_csv("GeminiVoterResultsandAns(4G,LLMVoter,Patched).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "LLMVoter Patched \n"+ str(eval(predans)))

# predans = pd.read_csv("GeminiVoterResultsandAns(G4,LLMVoter,Ex,Shuffle).csv")
# plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "LLMVoter \n"+ str(eval(predans)))


predans = pd.read_csv("AGU Work\Tests\GeminiVoterResultsandAns(4G,LLMVoter,Patched).csv")
print(eval(predans))
plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "MMPA (Balanced)")

predans = pd.read_csv("AGU Work\Tests\GeminiVoterResultsandAns(MCoTK).csv")
print(eval(predans))
plot_cmatrix(predans['Ground_Truth'], predans['Prediction Answer'], "MCoTK (Balanced)")
plt.show()