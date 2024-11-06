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
import random

def train_prediction_runAI(df, feature_columns, suffix):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Ground_Truth', axis=1), df['Ground_Truth'], test_size=0.66, random_state=42)
    # print(X_test.columns)
    # baseprompt = ""
    baseprompt = []
    print(y_train)
    trainfeatures = X_train[feature_columns]
    
    random.shuffle(baseprompt)
    baseprompt = ''.join(baseprompt)
    result = []
    prompts = []
    testfeatures = X_test[feature_columns]
    for rowind in range(0, len(X_test)):
        for column in feature_columns:
            result.append(testfeatures[column].iloc[rowind])
    print(X_test)
    X_test.insert(2, "Ground_Truth", y_test)
    X_test.insert(3, 'Decision_LLM'+suffix, result)
    X_test.to_csv("GeminiVoterResults(MCOTK).csv", index=False)

# df = pd.read_csv("GeminiEval(Categorical)(MTMP+)(6Shots,3Groups,150Samples)(Balanced).csv")
df = pd.read_csv("GeminiEval(Categorical)(MCoTK)(6Shots,1Group,150Samples)(Balanced).csv")
random.seed(42)
df = df.drop_duplicates(subset='Path')
print(len(df))

shotgroups = []
testgroups = []
# for i in range(0, len(df)):
#     row = df.iloc[i]
#     print(row['Path'])
# print(df.loc[df.iloc[255]['Path']])
# print(testsetpaths.iloc[0])
feature_columns = np.delete(df.columns,[0,1,2])
train_prediction_runAI(df, feature_columns, "46")
# print(feature_columns)