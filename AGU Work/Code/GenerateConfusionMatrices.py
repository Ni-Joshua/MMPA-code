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

def showConfusionMatrix(y_true, y_pred, title):
    print(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=["No", "New", "No New", "Similar", "Smaller", "Larger"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "New", "No New", "Similar", "Smaller", "Larger"])
    disp.plot()
    plt.title(title)
    plt.xlabel("Predicted Category")
    plt.ylabel("Ground Truth Category")

# df = pd.read_csv("Eval_Final\Test_Result(Balanced).csv")
# # for column in df.columns:
# #     if (column != 'Ground_Truth' and column != "Current_Day_FC" and column != 'Path'):
# #         print(column)
# #         precision = precision_score(df['Ground_Truth'], df[column], average='macro')
# #         recall = recall_score(df['Ground_Truth'], df[column], average='macro')
# #         f1 = f1_score(df['Ground_Truth'], df[column], average='macro')
# #         print(precision, recall, f1)
# for column in df.columns:
#     if (column != 'Ground_Truth' and column != "Current_Day_FC" and column != 'Path'):
#         name = column
#         if("_bs6" in column):
#             name = "Balanced MCoTK "
#         if("RF_b46" in column):
#             name = 'Balanced MTMP + RF'
#         showConfusionMatrix(df['Ground_Truth'], df[column], name)
# plt.show()

# df = pd.read_csv("AGU Work\Tests\GeminiVoterResultsandAns(4G,LLMVoter,Patched).csv")
# accuracy = accuracy_score(df['Ground_Truth'], df[column])
# precision = precision_score(df['Ground_Truth'], df[column], average='macro')
# recall = recall_score(df['Ground_Truth'], df[column], average='macro')
# f1 = f1_score(df['Ground_Truth'], df[column], average='macro')
# print(accuracy, precision, recall, f1)
# showConfusionMatrix(df['Ground_Truth'], df[column], "Balanced MTMP-A")
# plt.show()