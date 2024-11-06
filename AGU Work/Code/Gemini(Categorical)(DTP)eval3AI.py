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
import copy

genai.configure(api_key="")


# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
"temperature": 1,
"top_p": 0.95,
"top_k": 64,
"max_output_tokens": 8192,
"response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
model_name="gemini-1.5-pro",
generation_config=generation_config,
# safety_settings = Adjust safety settings
# See https://ai.google.dev/gemini-api/docs/safety-settings
)

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

def promptGenAI(rowind, features, isMainquestion):
    row = features.iloc[rowind]
    # options = []
    # for j in range(0, len(row)):
    #     if("Similar Active Fire" in row[j]):
    #         options.append("Similar Active Fire")
    #     elif("Smaller Active Fire" in row[j]):
    #         options.append("Smaller Active Fire")
    #     elif("Larger Active Fire" in row[j]):
    #         options.append("Larger Active Fire")
    #     elif("No New Active Fire" in row[j]):
    #         options.append("No New Active Fire")
    #     elif("No Active Fire" in row[j]):
    #         options.append("No Active Fire")
    #     elif("New Active Fire" in row[j]):
    #         options.append("New Active Fire")
    # options = set(options)
    # options = ', '.join(options)
    options = 'Similar Active Fire, Smaller Active Fire, Larger Active Fire, No New Active Fire, No Active Fire, New Active Fire'
    prompt = []
    # prompt.append("You will make the final decision on the evolution of a wildfire on the preceeding day. The following are the groups of variables analyzed and the final answers and explanations based on the variables. The explanations may not match the answers or variables so you must consider which answer makes the most sense. They are as follows: ")
    # for groupind in range(0, len(features.columns)):
    #     prompt.append("From the variables: " + features.columns[groupind].split("Group ")[1].split("_GeminiResponse")[0] + ", the answer and explanation are: " + row[groupind])
    # prompt.append("How will the wildfire evolve on the next day? The possible options for your answer are: "+options+ ". ")
    prompt.append("input: You will make the final decision on the evolution of a wildfire on the preceeding day. The following are the final decisions and explanations for the evolution of the wildfire after analyzing the designated variables.")
    for groupind in range(0, len(features.columns)):
        prompt.append("From the variables: " + features.columns[groupind].split("Group ")[1].split("_GeminiResponse")[0] + ", the answer and explanation are: " + row[groupind])
    if(isMainquestion == False):
        prompt.append("Given this information, how will the wildfire evolve on the next day? The possible options for your answer are: "+options+ ". Always provide an answer from the list. output: ")
    else:
        prompt.append("Given this information, how will the wildfire evolve on the next day? The possible options for your answer are: "+options+ ". Similar Active fire refers to a fire that has similar size to the current day, Smaller Active Fire refers to a fire that has a smaller size than the current day but there still is an existing fire, Larger Active Fire refers to a fire that has a larger size than the current day, No Active Fire refers to the current day's fire being completely extinguished, No New Active Fire refers to no new fire emerging, and New Active fire refers to a new fire emerging. Always provide an answer from the list. Please provide an explanation for your answer. output: ")
    prompt = ' '.join(prompt)
    return prompt

def getKnowledge():
    varData = {"VIIRS band M11": "Captures mid-infrared wavelengths. M11 is highly sensitive to thermal anomalies, making it effective for detecting active fires and identifying hotspots. It helps in early fire detection and monitoring fire spread.",
   "VIIRS band I2": "Captures near-infrared wavelengths. I2 is used to monitor vegetation health and stress, which can indicate areas of high fuel load and drought conditions. This information is crucial for assessing fire risk.",
   "VIIRS band I1": "Captures visible light, particularly in the red spectrum.I1 helps in mapping land cover and vegetation, providing insights into potential fuel sources. It also assists in post-fire assessment by highlighting burn scars.",
   "NDVI": "Measures live green vegetation using the difference between near-infrared (NIR) and red light. NDVI indicates vegetation health and density. Low NDVI values can signal drought-stressed vegetation, which is more susceptible to burning. High NDVI areas are less likely to ignite but can serve as fuel if they dry out.", 
   "EVI2": "Similar to NDVI but designed to be more sensitive in areas with high biomass. EVI2 provides a more accurate measure of vegetation vigor and canopy cover, particularly in dense forests. It helps in assessing the potential fuel load and fire behavior.", 
   "total precipitation": "Total amount of precipitation accumulated over a specific period. Precipitation data helps assess soil moisture and vegetation wetness. Low precipitation levels indicate dry conditions, increasing fire risk, while high precipitation can reduce fire danger by moistening fuels.", 
   "wind speed": "Rate of horizontal air movement. Wind speed is critical for predicting fire spread. High winds can rapidly spread fires by carrying embers and increasing the rate of fire movement.", 
   "wind direction": "Direction from which the wind is blowing. Wind direction helps in forecasting the potential path of a fire. Knowing wind direction is essential for predicting which areas are at risk and for planning evacuation routes and firefighting strategies.", 
   "minimum temperature": "Lowest temperature recorded over a specific period. Minimum temperatures affect overnight fire activity. Warmer nights can lead to active burning during nighttime, increasing overall fire risk.", 
   "maximum temperature": "Highest temperature recorded over a specific period. High maximum temperatures can dry out vegetation and fuels, increasing the likelihood of ignition and fire spread.", 
   "energy release component": "Index indicating the potential energy release per unit area in the flaming front of a fire. ERC is a critical indicator of fire intensity and potential difficulty in controlling a fire. Higher ERC values suggest more intense fires with higher energy release.", 
   "specific humidity": "Mass of water vapor per unit mass of air. Specific humidity affects fuel moisture content. Low specific humidity levels indicate dry air, which can dry out fuels and increase fire risk.", 
   "slope": "Steepness or incline of the terrain. Fires tend to spread more quickly uphill due to preheating of fuels above the fire. Steeper slopes can result in faster-moving and more intense fires.", 
   "aspect": "Compass direction that a slope faces. Aspect influences microclimates. South-facing slopes in the Northern Hemisphere receive more sunlight, tending to be drier and more fire-prone. Conversely, north-facing slopes may retain more moisture.", 
   "elevation": "Height above sea level. Elevation affects temperature and humidity. Higher elevations tend to be cooler and may have different vegetation types, influencing fire behavior.", 
   "Palmer drought severity index": "Measures the severity of drought conditions. PDSI provides a long-term perspective on moisture availability. Negative PDSI values indicate drought conditions, which can increase fire risk by drying out fuels.", 
   "landcover class": "Categories of the surface cover on the ground. Different land cover types (e.g., forest, grassland, urban areas) have varying fuel characteristics and fire behavior. Understanding land cover helps in assessing fire risk and planning firefighting strategies.", 
   "forecast total precipitation": "Predicted amount of precipitation for a future period. Forecast precipitation helps in anticipating changes in fuel moisture and potential fire activity. Wet conditions can reduce fire risk, while dry forecasts may indicate increased danger.", 
   "forecast wind speed": "Predicted rate of horizontal wind movement. Forecast wind speed is crucial for anticipating fire spread and intensity. High wind speeds can exacerbate fire conditions and complicate suppression efforts.", 
   "forecast wind direction": "Predicted direction from which the wind will blow. Forecast wind direction aids in predicting the potential path of a fire, helping in resource allocation and evacuation planning.", 
   "forecast temperature": "Predicted temperature for a future period. Forecast temperatures affect fuel dryness and fire behavior. High temperatures can increase fire risk and intensity.", 
   "forecast specific humidity": "Predicted specific humidity of the air. Forecast specific humidity provides insights into future moisture conditions. Low humidity forecasts suggest dry air, increasing fire risk.", 
   "current active fires": "Currently burning fires detected by satellite sensors. Active fire data is essential for real-time monitoring, assessing fire spread, and coordinating firefighting efforts."
   }
    kprompt = ["Here is some information about the variables analyzed:"]
    for key in varData.keys():
        kprompt.append(key + ": " + varData[key])
    kprompt.append("")
    return ' '.join(kprompt)



def train_prediction_runAI(df, feature_columns, suffix):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Ground_Truth', axis=1), df['Ground_Truth'], test_size=0.66, random_state=42)
    # print(X_test.columns)
    # baseprompt = ""
    baseprompt = []
    print(y_train)
    trainfeatures = X_train[feature_columns]
    # shotslocated = {
    # "Larger Active Fire": 0,
    # "Similar Active Fire": 0,
    # "Smaller Active Fire": 0,
    # "No Active Fire": 0,
    # "New Active Fire": 0,
    # "No New Active Fire": 0
    # }
    # for shotid  in range(0,len(y_train)):
    #     for key in shotslocated.keys():
    #         if(y_train.iloc[shotid] == key and shotslocated[key] <1):
    #             baseprompt = baseprompt+promptGenAI(shotid, trainfeatures, False)
    #             baseprompt = baseprompt + y_train.iloc[shotid] +'. '
    #             shotslocated[key]+=1
    #             break
    for shotid in range(0,5):
        # baseprompt = baseprompt+promptGenAI(shotid, trainfeatures, False)
        # baseprompt = baseprompt + y_train.iloc[shotid] +'. '
        baseprompt.append(promptGenAI(shotid, trainfeatures, False)+ y_train.iloc[shotid] +'. ')
    random.shuffle(baseprompt)
    baseprompt = ''.join(baseprompt)
    result = []
    prompts = []
    testfeatures = X_test[feature_columns]
    for rowind in range(0, len(X_test)):
        prompt = baseprompt + promptGenAI(rowind, testfeatures, True)
        prompts.append(prompt)
        for i in range(0,3):
            try:
                start = time.time()
                response = model.generate_content(prompt)
                end=time.time()
                print(f"Generate time taken: {end-start}")
                usage_metadata = response.usage_metadata
                print(f"Prompt Token Count: {usage_metadata.prompt_token_count}")
                # response = 'Done'
                response = ' '.join(response.text.splitlines())
                break
            except Exception as e:
                print(e)
                print("Retrying...")
                response = "Error"
        result.append(response)
        time.sleep(5)
    print(X_test)
    X_test.insert(2, "Ground_Truth", y_test)
    X_test.insert(3, 'Decision_LLM'+suffix, result)
    X_test.insert(2, 'Prompts', prompts)
    X_test.to_csv("GeminiVoterResults.csv", index=False)

def train_prediction_runAI_fixed(df, feature_columns, suffix):
    print(df)
    x = copy.deepcopy(df.drop("Ground_Truth", axis=1))
    y = copy.deepcopy(df['Ground_Truth'])
    baseprompt = []
    # trainfeatures = df[feature_columns]
    # shotslocated = {
    # "Larger Active Fire": 0,
    # "Similar Active Fire": 0,
    # "Smaller Active Fire": 0,
    # "No Active Fire": 0,
    # "New Active Fire": 0,
    # "No New Active Fire": 0
    # }
    # for shotid  in range(0,len(x)):
    #     found = 0
    #     allfound = False
    #     for key in shotslocated.keys():
    #         found+=shotslocated[key]
    #         if (found >=6):
    #             allfound = True
    #             break
    #         if(y.iloc[shotid] == key and shotslocated[key] <1):
    #             baseprompt.append(promptGenAI(shotid, trainfeatures, False)+ y.iloc[shotid] +'. ')
    #             shotslocated[key]+=1 
    #             x = x.drop([shotid], axis=0)
    #             print(len(x))
    #             y = y.drop([shotid], axis=0)
    #             break
    #     if allfound:
    #         break

    # random.shuffle(baseprompt)

    baseprompt = ''.join(baseprompt)
    result = []
    prompts = []
    testfeatures = x[feature_columns]
    for rowind in range(0, len(x)):
        prompt = baseprompt + promptGenAI(rowind, testfeatures, True)
        prompts.append(prompt)
        for i in range(0,3):
            try:
                start = time.time()
                response = model.generate_content(prompt)
                end=time.time()
                print(f"Generate time taken: {end-start}")
                usage_metadata = response.usage_metadata
                print(f"Prompt Token Count: {usage_metadata.prompt_token_count}")
                # response = 'Done'
                response = ' '.join(response.text.splitlines())
                break
            except Exception as e:
                print(e)
                print("Retrying...")
                response = "Error"
        result.append(response)
        time.sleep(5)
    print(x)
    x.insert(2, "Ground_Truth", y)
    x.insert(3, 'Decision_LLM_'+suffix, result)
    x.insert(2, 'Prompts', prompts)
    x.to_csv("GeminiVoterResults.csv(Fixed)(V2).csv", index=False)

# df = pd.read_csv("GeminiEval(Categorical)(MTMP+)(6Shots,3Groups,150Samples)(Balanced).csv")
df = pd.read_csv("GeminiEvalGeminiEval(ConsistencyTesting)(Main).csv")
random.seed(42)
# testsetpaths = pd.read_csv("Final Results\Test_Result(Balanced).csv")['Path']
# df = df.drop_duplicates(subset='Path')
print(len(df))

shotgroups = []
testgroups = []
# for i in range(0, len(df)):
#     row = df.iloc[i]
#     print(row['Path'])
# print(df.loc[df.iloc[255]['Path']])
# print(testsetpaths.iloc[0])
feature_columns = np.delete(df.columns,[0,1,2])
# train_prediction_runAI(df, feature_columns, "46")
train_prediction_runAI_fixed(df, feature_columns, "MTMP+")

# print(feature_columns)