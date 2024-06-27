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


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, average='weighted')
    # recall = recall_score(y_true, y_pred, average='weighted')
    # f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    # print(f"  Accuracy: {accuracy:.4f}")
    # print(f"  Precision: {precision:.4f}")
    # print(f"  Recall: {recall:.4f}")
    # print(f"  F1-score: {f1:.4f}")
    # print(np.mean([accuracy, precision, recall, f1]))
    # cm = confusion_matrix(y_true, y_pred, labels=["No", "New", "No_New", "Similar", "Smaller", "Larger"])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "New", "No_New", "Similar", "Smaller", "Larger"])
    # disp.plot()
    # plt.title(title)
    return [accuracy, precision, recall, f1]
    
def showConfusionMatrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["No", "New", "No_New", "Similar", "Smaller", "Larger"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "New", "No_New", "Similar", "Smaller", "Larger"])
    disp.plot()
    plt.title(title)
    # plt.show()

def transformData(unmodded, columns):
    df = []    
    for i in range(0, len(unmodded.index)):
        newrow = []
        row = unmodded.iloc[i]
        newrow.append(row["Path"])
        newrow.append(row["Current_Day_Fire_Count"])
        # fc = int(row["Current_Day_Fire_Count"])
        # if (fc == 0):
        #     newrow.append("0")
        # elif (fc <= 10):
        #     newrow.append("1-10")
        # elif (fc <= 50):
        #     newrow.append("11-50")
        # elif (fc <= 100):
        #     newrow.append("51-100")
        # elif (fc <= 500):
        #     newrow.append("101-500")
        # else:
        #     newrow.append("500+") 
        for j in range(3, len(row)):
            if("Similar Active Fire" in row[j]):
                newrow.append("Similar")
            elif("Smaller Active Fire" in row[j]):
                newrow.append("Smaller")
            elif("Larger Active Fire" in row[j]):
                newrow.append("Larger")
            elif("No New Active Fire" in row[j]):
                newrow.append("No_New")
            elif("No Active Fire" in row[j]):
                newrow.append("No")
            elif("New Active Fire" in row[j]):
                newrow.append("New")
            else:
                newrow.append("No Answer")
        if("Similar Active Fire" in row["Ground_Truth"]):
            newrow.append("Similar")
        elif("Smaller Active Fire" in row["Ground_Truth"]):
            newrow.append("Smaller")
        elif("Larger Active Fire" in row["Ground_Truth"]):
            newrow.append("Larger")
        elif("No New Active Fire" in row["Ground_Truth"]):
            newrow.append("No_New")
        elif("No Active Fire" in row["Ground_Truth"]):
            newrow.append("No")
        elif("New Active Fire" in row["Ground_Truth"]):
            newrow.append("New")
        df.append(newrow)
    df = np.array(df)
    df = pd.DataFrame(df, columns = columns)
    return df

def train_prediction(df, feature_columns, sufix, title):
    test_results, train_results = train_prediction_run(df, feature_columns, sufix, title, "")
    return test_results, train_results

    # grouped = df.groupby('Ground_Truth')
    # nofire = [grouped.get_group('No_New'), grouped.get_group('New')]
    # fire = [grouped.get_group('Larger'),  grouped.get_group('Similar'), grouped.get_group('Smaller'), grouped.get_group('No')]

    # nofire_df = pd.concat(nofire)
    # fire_df = pd.concat(fire)

    # #breakdowns

    # nofire_test_result, nofire_train_result = train_prediction_run(nofire_df, feature_columns, sufix+"_Breakdown", title, "_No_Fire")
    # fire_test_result, fire_train_result = train_prediction_run(fire_df, feature_columns, sufix+"_Breakdown", title, "_Fire")
    # test_result = pd.concat([nofire_test_result, fire_test_result])
    # train_result = pd.concat([nofire_train_result, fire_train_result])
    # return test_result, train_result


def train_prediction_run(df, feature_columns, sufix, title, breakdown):
    df = df[feature_columns]
    dummy_columns = df.columns.drop(['Path', 'Ground_Truth', 'Current_Day_FC'])
    # dummy_columns = df.columns.drop(['Path', 'Ground_Truth'])
    df_encoded = pd.get_dummies(df, columns=dummy_columns)
    # Separate features and target variable
    X = df_encoded.drop('Ground_Truth', axis=1)
    y = df_encoded['Ground_Truth']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    train_path = X_train.pop('Path')
    test_path = X_test.pop('Path')

    # Instantiate models
    dt_model = DecisionTreeClassifier()
    rf_model = RandomForestClassifier()
    lr_model = LogisticRegression()
    svm_model = SVC()

    # Fit models
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # Save models
    with open(f"{title}{sufix}{breakdown}_DT.pkl", 'wb') as f:
        pickle.dump(dt_model, f)
    with open(f"{title}{sufix}{breakdown}_RF.pkl", 'wb') as f:
        pickle.dump(rf_model, f)
    with open(f"{title}{sufix}{breakdown}_LR.pkl", 'wb') as f:
        pickle.dump(lr_model, f)
    with open(f"{title}{sufix}{breakdown}_SVM.pkl", 'wb') as f:
        pickle.dump(svm_model, f)

    #predictions
    y_test_dt = dt_model.predict(X_test)
    y_test_rf = rf_model.predict(X_test)
    y_test_lr = lr_model.predict(X_test)
    y_test_svm = svm_model.predict(X_test)

    y_train_dt = dt_model.predict(X_train)
    y_train_rf = rf_model.predict(X_train)
    y_train_lr = lr_model.predict(X_train)
    y_train_svm = svm_model.predict(X_train)

    # # Evaluate models
    # showConfusionMatrix(y_test, y_test_dt, f"{title} Decision Tree")
    # showConfusionMatrix(y_test, y_test_rf, f"{title} Random Forest")
    # showConfusionMatrix(y_test, y_test_lr, f"{title} Logistic Regression")
    # showConfusionMatrix(y_test, y_test_svm, f"{title} SVM")

    # Merge the results

    test_result = pd.DataFrame({'Path': test_path, 'Decision_DT'+sufix: y_test_dt, 'Decision_RF'+sufix: y_test_rf, 'Decision_LR'+sufix: y_test_lr, 'Decision_SVM'+sufix: y_test_svm})
    train_result = pd.DataFrame({'Path': train_path, 'Decision_DT'+sufix: y_train_dt, 'Decision_RF'+sufix: y_train_rf, 'Decision_LR'+sufix: y_train_lr, 'Decision_SVM'+sufix: y_train_svm})
    return test_result, train_result

def generateMergedDatasetB():
    b454Columns = ["Path", "Current_Day_FC","Fuel_b454", "Moisture_b454", "Topology/Geography_b454", "Weather_b454", 'Ground_Truth']
    b46Columns = ["Path", "Current_Day_FC","Fuel_b46", "Moisture_b46", "Topology/Geography_b46", "Weather_b46", 'Ground_Truth']
    b40Columns = ["Path", "Current_Day_FC","Fuel_b40", "Moisture_b40", "Topology/Geography_b40", "Weather_b40", 'Ground_Truth']
    b16Columns = ["Path", "Current_Day_FC", "Prediction_b16", 'Ground_Truth']
    b10Columns = ["Path", "Current_Day_FC", "Prediction_b10", 'Ground_Truth']
    bs6Columns = ["Path", "Current_Day_FC", "Prediction_bs6", 'Ground_Truth']
    bs0Columns = ["Path", "Current_Day_FC", "Prediction_bs0", 'Ground_Truth']

    b454 = transformData(pd.read_csv("GeminiEval(Categorical)(DTP)(54Shots,4Groups,300Samples)(Balanced).csv"), b454Columns).drop(columns=["Current_Day_FC"])
    b454 = b454.drop_duplicates(subset='Path')
    b46 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Balanced/4 Groups/6 Shots/GeminiEval(Categorical)(DTP)(6Shots,4Groups,300Samples)(Balanced).csv"), b46Columns)
    b46 = b46.drop_duplicates(subset='Path')
    b40 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Balanced/4 Groups/0 Shots/GeminiEval(Categorical)(DTP)(0Shots,4Groups,300Samples)(Balanced).csv"), b40Columns).drop(columns=["Current_Day_FC"])
    b40 = b40.drop_duplicates(subset='Path')
    b16 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Balanced/1 Group/6 Shots/GeminiEval(Categorical)(DTP)(6Shots,1Group,300Samples)(Balanced).csv"), b16Columns).drop(columns=["Current_Day_FC"])
    b16 = b16.drop_duplicates(subset='Path')
    b10 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Balanced/1 Group/0 Shots/GeminiEval(Categorical)(DTP)(0Shots,1Group,300Samples)(Balanced).csv"), b10Columns).drop(columns=["Current_Day_FC"])
    b10 = b10.drop_duplicates(subset='Path')
    bs6 = transformData(pd.read_csv("New Run 6-22-2024/SP Results/Balanced/1 Group/6 Shots/GeminiEval(Categorical)(SP)(6Shots,1Group,300Samples)(Balanced).csv"), bs6Columns).drop(columns=["Current_Day_FC"]) 
    bs6 = bs6.drop_duplicates(subset='Path')
    bs0 = transformData(pd.read_csv("New Run 6-22-2024/SP Results/Balanced/1 Group/0 Shots/GeminiEval(Categorical)(SP)(0Shots,1Group,300Samples)(Balanced).csv"), bs0Columns).drop(columns=["Current_Day_FC"])
    bs0 = bs0.drop_duplicates(subset='Path')

    dataFramesB = [ b46, b40, b16, b10, bs6, bs0, b454,]
    df_mergedB = reduce(lambda  left,right: pd.merge(left,right,on=['Path', 'Ground_Truth'], how='inner'), dataFramesB)
    df_mergedB.to_csv("MergedLLMOutputB.csv", index=False)

    return df_mergedB

def generateMergedDatasetN():
    n46Columns = ["Path", "Current_Day_FC","Fuel_n46", "Moisture_n46", "Topology/Geography_n46", "Weather_n46", 'Ground_Truth']
    n40Columns = ["Path", "Current_Day_FC","Fuel_n40", "Moisture_n40", "Topology/Geography_n40", "Weather_n40", 'Ground_Truth']
    n16Columns = ["Path", "Current_Day_FC", "Prediction_n16", 'Ground_Truth']
    n10Columns = ["Path", "Current_Day_FC", "Prediction_n10", 'Ground_Truth']
    ns6Columns = ["Path", "Current_Day_FC", "Prediction_ns6", 'Ground_Truth']
    ns0Columns = ["Path", "Current_Day_FC", "Prediction_ns0", 'Ground_Truth']
    n46 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Natural/4 Groups/6 Shots/GeminiEval(Categorical)(DTP)(6Shots,4Groups,300Samples)(Natural).csv"), n46Columns)
    n46 = n46.drop_duplicates(subset='Path')
    n40 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Natural/4 Groups/0 Shots/GeminiEval(Categorical)(DTP)(0Shots,4Groups,300Samples)(Natural).csv"), n40Columns).drop(columns=["Current_Day_FC"])
    n40 = n40.drop_duplicates(subset='Path')
    n16 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Natural/1 Group/6 Shots/GeminiEval(Categorical)(DTP)(6Shots,1Group,300Samples)(Natural).csv"), n16Columns).drop(columns=["Current_Day_FC"])
    n16 = n16.drop_duplicates(subset='Path')
    n10 = transformData(pd.read_csv("New Run 6-22-2024/DTP Results/Natural/1 Group/0 Shots/GeminiEval(Categorical)(DTP)(0Shots,1Group,300Samples)(Natural).csv"), n10Columns).drop(columns=["Current_Day_FC"])
    n10 = n10.drop_duplicates(subset='Path')
    ns6 = transformData(pd.read_csv("New Run 6-22-2024/SP Results/Natural/1 Group/6 Shots/GeminiEval(Categorical)(SP)(6Shots,1Group,300Samples)(Natural).csv"), ns6Columns).drop(columns=["Current_Day_FC"])
    ns6 = ns6.drop_duplicates(subset='Path')
    ns0 = transformData(pd.read_csv("New Run 6-22-2024/SP Results/Natural/1 Group/0 Shots/GeminiEval(Categorical)(SP)(0Shots,1Group,300Samples)(Natural).csv"), ns0Columns).drop(columns=["Current_Day_FC"])
    ns0 = ns0.drop_duplicates(subset='Path')

    dataFramesN = [n46, n40, n16, n10, ns6, ns0]
    df_mergedN = reduce(lambda  left,right: pd.merge(left,right,on=['Path', 'Ground_Truth'], how='outer'), dataFramesN)
    df_mergedN.to_csv("MergedLLMOutputN.csv", index=False)

def generateMergedDataset():
    df_mergedN = generateMergedDatasetN()
    df_mergedB = generateMergedDatasetB()
    df_merged = pd.merge(df_mergedN,df_mergedB,on=['Path'], how='outer')
    df_merged.to_csv("MergedLLMOutput.csv", index=False)

def evaluateAllColumns(df, prefix, title):
    results = []
    for column in df.columns:
        if (column != 'Ground_Truth'):
            result = [column]
            result.extend(evaluate(df['Ground_Truth'], df[column]))
            results.append(result)
            if title is not None:
                showConfusionMatrix(df['Ground_Truth'], df[column], title + " " + column)
    results = np.array(results)
    results = pd.DataFrame(results, columns = ["Column", prefix+"Accuracy", prefix+"Precision", prefix+"Recall", prefix+"F1"])
    return results

def dataAnalyze(df, title):
    df = df.drop(columns=['Path', 'Current_Day_FC'])
    gt = df.pop('Ground_Truth')
    df.insert(0, 'Ground_Truth', gt)
    print(df.columns)

    #data distrubtions 
    vc = df.apply(pd.Series.value_counts)
    vc.plot.bar(figsize=(20,10))
    plt.title(title)
    plt.show()

    #evaluation per raw prediction
    evaluateAllColumns(df, "", title).to_csv(f"EvalResults({title}).csv", index=False)    

    #evaluation per category per raw prediction
    df_breakdowns = []
    grouped = df.groupby('Ground_Truth')
    nofire = [grouped.get_group('No_New'), grouped.get_group('New')]
    fire = [grouped.get_group('Larger'),  grouped.get_group('Similar'), grouped.get_group('Smaller'), grouped.get_group('No')]
    fireDf = pd.concat(fire)
    nofireDf = pd.concat(nofire)
    fireDf = evaluateAllColumns(fireDf, "Fire_", None)
    nofireDf = evaluateAllColumns(nofireDf, "No_Fire_", None)
    df_breakdowns = fireDf.merge(nofireDf, on='Column')
    # for name, group in grouped:
    #     groupDf = evaluateAllColumns(group, name+"_", None)
    #     df_breakdowns.append(groupDf)
    # df_breakdowns = reduce(lambda  left,right: pd.merge(left,right,on=['Column']), df_breakdowns)
    df_breakdowns.to_csv(f"EvalResultsBreakdown({title}).csv", index=False)

def train_prediction_balanced(df):

    title = "Balanced"
    all_columns = df.columns
    df_test_result, df_train_result = train_prediction(df, all_columns, "_All", title)
    
    # two_panel_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_b46', 'Moisture_b46','Topology/Geography_b46', 'Weather_b46', 'Fuel_b40', 'Moisture_b40', 'Topology/Geography_b40', 'Weather_b40']
    # test_result, train_result = train_prediction(df, two_panel_columns, "_Two_Panel", title)
    # df_test_result = df_test_result.merge(test_result, on='Path')
    # df_train_result = df_train_result.merge(train_result, on='Path')

    b46_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_b46', 'Moisture_b46', 'Topology/Geography_b46', 'Weather_b46']
    test_result, train_result = train_prediction(df, b46_columns, "_b46", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')

    b40_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_b40', 'Moisture_b40', 'Topology/Geography_b40', 'Weather_b40']
    test_result, train_result = train_prediction(df, b40_columns, "_b40", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')

    # b4616_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_b46', 'Moisture_b46', 'Topology/Geography_b46', 'Weather_b46','Prediction_b16']
    # test_result, train_result = train_prediction(df, b4616_columns, "_b4616", title)
    # df_test_result = df_test_result.merge(test_result, on='Path')
    # df_train_result = df_train_result.merge(train_result, on='Path')


    b4610_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_b46', 'Moisture_b46', 'Topology/Geography_b46', 'Weather_b46','Prediction_b10']
    test_result, train_result = train_prediction(df, b4610_columns, "_b4610", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')

    b4010_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_b40', 'Moisture_b40', 'Topology/Geography_b40', 'Weather_b40','Prediction_b10']
    test_result, train_result = train_prediction(df, b4010_columns, "_b4010", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')

    
    b454_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_b454', 'Moisture_b454', 'Topology/Geography_b454', 'Weather_b454']
    test_result, train_result = train_prediction(df, b454_columns, "_b454", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')

    df_test_result.merge(df, on='Path').to_csv(f"Test_Result({title}).csv", index=False)
    df_train_result.merge(df, on='Path').to_csv(f"Train_Result({title}).csv", index=False)
    return df_test_result, df_train_result

def train_prediction_natural(df):
    
    title = "Natural"
    all_columns = df.columns
    df_test_result, df_train_result = train_prediction(df, all_columns, "_All", title)
            
    # two_panel_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_n46', 'Moisture_n46','Topology/Geography_n46', 'Weather_n46', 'Fuel_n40', 'Moisture_n40', 'Topology/Geography_n40', 'Weather_n40']
    # test_result, train_result = train_prediction(df, two_panel_columns, "_Two_Panel", title)
    # df_test_result = df_test_result.merge(test_result, on='Path')
    # df_train_result = df_train_result.merge(train_result, on='Path')
        
    n46_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_n46', 'Moisture_n46', 'Topology/Geography_n46', 'Weather_n46']
    test_result, train_result = train_prediction(df, n46_columns, "_n46", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')
        
    n40_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_n40', 'Moisture_n40', 'Topology/Geography_n40', 'Weather_n40']
    test_result, train_result = train_prediction(df, n40_columns, "_n40", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')
        
    n4610_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_n46', 'Moisture_n46', 'Topology/Geography_n46', 'Weather_n46','Prediction_n10']
    test_result, train_result = train_prediction(df, n4610_columns, "_n4610", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')
        
    n4010_columns = ['Path', 'Ground_Truth', 'Current_Day_FC', 'Fuel_n40', 'Moisture_n40', 'Topology/Geography_n40', 'Weather_n40','Prediction_n10']
    test_result, train_result = train_prediction(df, n4010_columns, "_n4010", title)
    df_test_result = df_test_result.merge(test_result, on='Path')
    df_train_result = df_train_result.merge(train_result, on='Path')

    df_test_result.merge(df, on='Path').to_csv(f"Test_Result({title}).csv", index=False)
    df_train_result.merge(df, on='Path').to_csv(f"Train_Result({title}).csv", index=False)
    return df_test_result, df_train_result


def evalBalanced():
    dfB = pd.read_csv("MergedLLMOutputB.csv")
    train_prediction_balanced(dfB)
    # dataAnalyze(dfB, "Balanced")
    df_test_result = pd.read_csv("Test_Result(Balanced).csv")
    dataAnalyze(df_test_result, "Balanced Test")
    plt.show()
    # df_train_result = pd.read_csv("Train_Result(Balanced).csv")
    # dataAnalyze(df_train_result, "Balanced Train")

def evalNatural():
    dfN = pd.read_csv("MergedLLMOutputN.csv")
    train_prediction_natural(dfN)
    # dataAnalyze(dfN, "Natural")
    df_test_result = pd.read_csv("Test_Result(Natural).csv")
    dataAnalyze(df_test_result, "Natural Test")
    plt.show()
    # df_train_result = pd.read_csv("Train_Result(Natural).csv")
    # dataAnalyze(df_train_result, "Natural Train")

generateMergedDatasetB()
evalBalanced()

generateMergedDatasetN()
evalNatural()

