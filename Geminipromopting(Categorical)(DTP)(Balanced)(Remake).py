import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import os
from PIL import Image
import google.generativeai as genai
import pandas as pd
import copy
import random
import time
import json
import time

genai.configure(api_key="")
random.seed(24)


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
model_name="gemini-1.5-flash",
generation_config=generation_config,
# safety_settings = Adjust safety settings
# See https://ai.google.dev/gemini-api/docs/safety-settings
)

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  start = time.time()
  file = genai.upload_file(path, mime_type=mime_type)
  end = time.time()
  print(f"Uploaded file '{file.display_name}' as: {file.uri} in {end-start} seconds")
  return file
#   return path



# varData = {    "VIIRS band M11": "Map of surface reflectance, may be beneficial for distinguishing health vegetation from clouds and smoke.",
#                "VIIRS band I2": "Map of surface reflectance, may be beneficial for distinguishing health vegetation from clouds and smoke.",
#                "VIIRS band I1": "Map of surface reflectance, may be beneficial for distinguishing health vegetation from clouds and smoke.",
#                "NDVI": "Vegetation index that may provide more infomation about living fuel",
#                "EVI2": "Vegetation index that may provide more infomation about living fuel",
#                "total precipitation": "Amount of rainfall, may be beneficial for determing moisture in the soil and air",
#                "wind speed": "Speed of the wind, may be beneficial for determing amount of oxygen given to the fire",
#                "wind direction": "Direction of the wind",
#                "minimum temperature": "Lowest temperature measured",
#                "maximum temperature": "Highest temperature measured",
#                "energy release component": "Energy released during the combustion of the material in the area",
#                "specific humidity": "Amount of water contained in an amount of air, the smaller the number, the less water and thus the area may be more prone to fires.",
#                "slope": "Angle of the land, fires are likely to move faster up slopes",
#                "aspect": "Compass orientation of the slope",
#                "elevation": "Level above sea level, higher elevations have less oxygen",
#                "Palmer drought severity index": "Index that determines how dry the area is, the dryer the area, the more likely there will be a fire. A smaller number represents a more dry area.",
#                "landcover class": "The fuel type of the area. The color in the pixel represents the type of land cover class. The number to area categories are as follows: 1: Evergreem Needleleaf Forests, 2: Evergreen Broadleaf Forests, 3: Deciduous Needleleaf Forests, 4: Deciduous Broadleaf Forests, 5: Mixed Forests, 6: Closed Shrublands, 7: Open Shrublands, 8: Woody Savannas, 9: Savannas, 10: Grasslands, 11: Permanent Wetlands, 12: Croplands, 13: Urban and Built-up lands, 14: Cropland/Natural Vegetation Mosaics, 15: Permanent Snow and Ice, 16: Barren, 17: Water Bodies.",
#                "forecast total precipitation": "Predicted amount of preciptiation on the next day",
#                "forecast wind speed": "Predicted speed of the wind on the next day",
#                "forecast wind direction": "Predicted direction of the wind on the next day",
#                "forecast temperature": "Predicted temperature of the next day",
#                "forecast specific humidity": "Predicted amount of water in the air on the next day",
#                "current active fires": "Every pixel that is not purple represents an active fire. A lighter color for current active fire pixels represents a more recent detection time."
#                }
# varData = {"VIIRS band M11": "The Moderate Resolution Band 11 (M11) on the VIIRS sensor captures mid-infrared wavelengths. This band is particularly useful for detecting thermal anomalies, such as wildfires, and for soil and vegetation analysis. It can also help in identifying surface temperature and moisture content.",
#  "VIIRS band I2": "The Imagery Resolution Band 2 (I2) on the VIIRS sensor captures near-infrared wavelengths. Near-infrared data is crucial for monitoring vegetation health, water bodies, and soil moisture. It is commonly used in vegetation indices like NDVI", 
#  "VIIRS band I1": "The Imagery Resolution Band 1 (I1) on the VIIRS sensor captures visible light, particularly in the red spectrum. This band is used for general imaging, vegetation analysis, and mapping. It helps in distinguishing different types of land cover.", 
#  "NDVI": "NDVI is a measure of live green vegetation using the difference between near-infrared (NIR) and red light reflected by vegetation. NDVI values range from -1 to 1, with higher values indicating healthier and denser vegetation. It is widely used for monitoring plant health, drought, and crop conditions.", 
#  "EVI2": "EVI2 is similar to NDVI but uses two spectral bands and is designed to improve sensitivity in areas with high biomass. EVI2 adjusts for atmospheric conditions and canopy background noise, providing a more accurate measure of vegetation cover and vigor, especially in dense forested areas.", 
#  "total precipitation": "The total amount of precipitation (rain, snow, sleet, etc.) accumulated over a specific period, typically measured in millimeters (mm). This measure is crucial for hydrological studies, agricultural planning, and weather forecasting.", 
#  "wind speed": "The rate at which air is moving horizontally past a certain point, usually measured in meters per second (m/s) or kilometers per hour (km/h). Wind speed is vital for weather prediction, aviation, marine activities, and assessing potential fire spread.", 
#  "wind direction": "The direction from which the wind is blowing, typically measured in degrees from true north. Knowing wind direction is essential for meteorology, navigation, and understanding pollutant dispersion.", 
#  "minimum temperature": "The lowest temperature recorded over a specified period, typically within a day. Minimum temperature data is important for agricultural planning, assessing frost risk, and climate studies.", 
#  "maximum temperature": "The highest temperature recorded over a specified period, typically within a day. Maximum temperature is used in weather forecasting, heatwave monitoring, and climate analysis.", 
#  "energy release component": "An index used in fire behavior prediction models, indicating the potential energy release per unit area in the flaming front of a fire. Higher ERC values suggest more intense fires. It helps in wildfire risk assessment and management.", 
#  "specific humidity": "The mass of water vapor per unit mass of air, usually measured in grams per kilogram (g/kg). This measure provides an absolute indicator of moisture content in the air, important for weather prediction and understanding humidity conditions.", 
#  "slope": "The steepness or incline of the terrain, often measured in degrees or as a percentage. Slope data is crucial for topographic mapping, landslide risk assessment, and hydrological modeling.", 
#  "aspect": "The compass direction that a slope faces, typically measured in degrees from north. Aspect affects microclimates, vegetation growth, and erosion patterns. It is used in environmental and geological studies.", 
#  "elevation": "The height above a reference point, typically sea level, measured in meters or feet. Elevation data is essential for topography, climate modeling, and infrastructure development.", 
#  "Palmer drought severity index": "A metric for measuring the severity of drought conditions, taking into account temperature, precipitation, and soil moisture. Positive values indicate wet conditions, while negative values indicate dry conditions. PDSI is widely used for drought monitoring and water resource management.", 
#  "landcover class": "Categories of the surface cover on the ground, such as forest, grassland, urban areas, etc., usually determined from satellite imagery. Landcover classification is used in environmental monitoring, urban planning, and resource management. The color in the pixel represents the type of land cover class. The number to area categories are as follows: 1: Evergreem Needleleaf Forests, 2: Evergreen Broadleaf Forests, 3: Deciduous Needleleaf Forests, 4: Deciduous Broadleaf Forests, 5: Mixed Forests, 6: Closed Shrublands, 7: Open Shrublands, 8: Woody Savannas, 9: Savannas, 10: Grasslands, 11: Permanent Wetlands, 12: Croplands, 13: Urban and Built-up lands, 14: Cropland/Natural Vegetation Mosaics, 15: Permanent Snow and Ice, 16: Barren, 17: Water Bodies.", 
#  "forecast total precipitation": "The predicted amount of precipitation for a future period. This helps in weather forecasting, flood prediction, and agricultural planning.", 
#  "forecast wind speed": "The predicted rate of horizontal wind movement for a future period. Forecast wind speed is important for weather forecasting, aviation, and assessing potential fire spread.",
#  "forecast wind direction": "The predicted direction from which the wind will blow for a future period. This is used in meteorology, navigation, and understanding pollutant dispersion.", 
#  "forecast temperature": "The predicted temperature for a future period, often including both minimum and maximum values. Forecast temperature is crucial for weather prediction, agricultural planning, and climate studies.", 
#  "forecast specific humidity": "The predicted specific humidity (moisture content) of the air for a future period. This helps in weather forecasting and understanding humidity conditions.", 
#  "current active fires": "Refers to currently burning fires detected by satellite sensors, often indicated by thermal anomalies or hotspots in imagery. Active fire data is used in wildfire monitoring, emergency response, and environmental impact assessment. Every pixel that is not purple represents an active fire. A lighter color for current active fire pixels represents a more recent detection time."
#  }
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
   "landcover class": "Categories of the surface cover on the ground. Different land cover types (e.g., forest, grassland, urban areas) have varying fuel characteristics and fire behavior. Understanding land cover helps in assessing fire risk and planning firefighting strategies. The color in the pixel represents the type of land cover class. The number to area categories are as follows: 1: Evergreem Needleleaf Forests, 2: Evergreen Broadleaf Forests, 3: Deciduous Needleleaf Forests, 4: Deciduous Broadleaf Forests, 5: Mixed Forests, 6: Closed Shrublands, 7: Open Shrublands, 8: Woody Savannas, 9: Savannas, 10: Grasslands, 11: Permanent Wetlands, 12: Croplands, 13: Urban and Built-up lands, 14: Cropland/Natural Vegetation Mosaics, 15: Permanent Snow and Ice, 16: Barren, 17: Water Bodies.", 
   "forecast total precipitation": "Predicted amount of precipitation for a future period. Forecast precipitation helps in anticipating changes in fuel moisture and potential fire activity. Wet conditions can reduce fire risk, while dry forecasts may indicate increased danger.", 
   "forecast wind speed": "Predicted rate of horizontal wind movement. Forecast wind speed is crucial for anticipating fire spread and intensity. High wind speeds can exacerbate fire conditions and complicate suppression efforts.", 
   "forecast wind direction": "Predicted direction from which the wind will blow. Forecast wind direction aids in predicting the potential path of a fire, helping in resource allocation and evacuation planning.", 
   "forecast temperature": "Predicted temperature for a future period. Forecast temperatures affect fuel dryness and fire behavior. High temperatures can increase fire risk and intensity.", 
   "forecast specific humidity": "Predicted specific humidity of the air. Forecast specific humidity provides insights into future moisture conditions. Low humidity forecasts suggest dry air, increasing fire risk.", 
   "current active fires": "Currently burning fires detected by satellite sensors. Active fire data is essential for real-time monitoring, assessing fire spread, and coordinating firefighting efforts. Every pixel that is not purple represents an active fire. A lighter color for current active fire pixels represents a more recent detection time."
   }

questions = {  "Which Categories": "Will the next day have a Similar Active Fire, a Larger Active Fire, a Smaller Active Fire, or No Active Fire? If there are no current active fires, will the next day have No New Active Fire or will it have a New Active Fire? Answer with these options: Similar Active Fire, Larger Active Fire, Smaller Active Fire, No Active Fire, New Active Fire, No New Active Fire. Think step by step and provide the reasoning for your decision",
               "New Fire": "Please predict if there will be a new wildfire or not on the next day only using the information in the above images. Answer the question with the following options: “New Active Fire” or “No New Active Fire”. Think step by step and explain your reasoning following the answer.",
               "Spread":"Please predict if the fire on the next day is larger, smaller, or similar in size compared to the current day only using the information in the above images.  Answer the question with the following options: Similar Active Fire, Larger Active Fire or Smaller Active Fire. Think step by step and explain your reasoning following the answer.",
               "No Fire": "The fire is decreasing in size. Please predict if there is an active fire on the next day only using the information in the above images.  Answer the question with the following options: No Active Fire, or Smaller Active Fire. Think step by step and explain your reasoning following the answer."}
             
def buildFireData(varData, currentDayDirectory):
   fireData = {}
   fileList = os.listdir(currentDayDirectory)
   for file in fileList:
      if("shrunk" in file or "expand" in file or "shift" in file):
         continue
      varName = file.split("_")[0]
      if(varName == 'active fires'):
         varName = 'current active fires'
      fireData[varName] = {}
      fireData[varName]["info"] = varData[varName]
      fireData[varName]["minVal"] = file.split("range(")[-1].split(")")[0].split(',')[0]
      fireData[varName]["maxVal"] = file.split("range(")[-1].split(")")[0].split(',')[1]
      fireData[varName]['path'] = os.path.join(currentDayDirectory, file)
      if "current active fires" in varName:
         fireCount = file.split("_fc(")[-1].split(")")[0]
         fireData[varName]["fireCount"] = int(fireCount)   
   return fireData

def calculateGroundTruth(varData, currentDayFireData, nextDayFireData):
   # print(currentDayFireData['current active fires']['fireCount'])
   # print(nextDayFireData['current active fires']['fireCount'])
   currentFireCount = currentDayFireData["current active fires"]["fireCount"]
   nextFireCount = nextDayFireData["current active fires"]["fireCount"]
   if(currentFireCount == 0 and nextFireCount == 0):
      return "No New Active Fire"
   if nextFireCount == 0:
      return "No Active Fire"
   if currentFireCount == 0 and nextFireCount > 0:
      return "New Active Fire"
   if max(nextFireCount, currentFireCount) * 0.9 <= min(nextFireCount, currentFireCount):
      return "Similar Active Fire"
   if nextFireCount > currentFireCount:
      return "Larger Active Fire"
   if nextFireCount < currentFireCount:
      return "Smaller Active Fire"

def createVarGroupPrompt(varGroup, fireData):
   prompt = []
   prompt.append("input: Here is a list of images relevant to the wildfire. They are captured in the same area and day. The forecast variables represent predictions of the variable on the following day. A color closer to purple represents a smaller number and a color closer to yellow represents a larger number. Each pixel represents a 375m by 375m area.")
   for varName in varGroup:
         prompt.append(varName +". " + fireData[varName]["info"] + " The values, indicated by the color, range from " + fireData[varName]["minVal"] +" to "+ fireData[varName]["maxVal"] + ".")
         # prompt.append(varName +", Range: " + fireData[varName]["minVal"] +" to "+ fireData[varName]["maxVal"])
         prompt.append(upload_to_gemini(fireData[varName]["path"], mime_type="image/png"))
   return prompt

def createSample(varData, varGroups, currentDayDirectory, nextDayDirectory, sampleBal):
   currentDayFireData = buildFireData(varData, currentDayDirectory)
   nextDayFireData = buildFireData(varData, nextDayDirectory)
   groundTruth = calculateGroundTruth(varData, currentDayFireData, nextDayFireData)
   sample  = { "currentDayFireData": currentDayFireData,
                "nextDayFireData": nextDayFireData,
                "groundTruth": groundTruth}
   if(sampleBal[groundTruth] <= 0):
      return None
   else:
      sampleBal[groundTruth] -= 1
   for varGroup in varGroups:
      varGroupKey = " ".join(varGroup)
      prompt = createVarGroupPrompt(varGroup, currentDayFireData)
      sample[varGroupKey] = prompt
   return sample

def createSamples(varData, varGroups, rootDir, sampleBal):
   samples = []
   while(sampleBal['Larger Active Fire'] + sampleBal['Smaller Active Fire'] + sampleBal['Similar Active Fire'] + sampleBal['No Active Fire']+ sampleBal['New Active Fire'] +sampleBal["No New Active Fire"]> 0):
      yearFolder = os.listdir(rootDir)[random.randint(0, len(os.listdir(rootDir))-1)]
      yearDirectory = os.path.join(rootDir, yearFolder)
      fireFolder = os.listdir(yearDirectory)[random.randint(0, len(os.listdir(yearDirectory))-1)]
      fireDirectory = os.path.join(yearDirectory, fireFolder)
      dayid = random.randint(0, len(os.listdir(fireDirectory))-2)
      currentDayFolder = os.listdir(fireDirectory)[dayid]
      nextDayFolder = os.listdir(fireDirectory)[dayid+1]
      currentDayDirectory = os.path.join(fireDirectory, currentDayFolder)
      followingDayDirectory = os.path.join(fireDirectory, nextDayFolder)
      sample = createSample(varData, varGroups, currentDayDirectory, followingDayDirectory, sampleBal)
      if(not sample is None):
         samples.append(sample)
   return samples

def appendToPrompt(prompt, sample, varGroup, question):
   varGroupKey = " ".join(varGroup)
   prompt.extend(sample[varGroupKey])
   prompt.append(question)
   prompt.append("output: ")

def generateResponse(trainingSamples, testingSample, varGroup, question):
   prompt = []
   # for key in varData.keys():
   #    prompt.append(key + ": " + varData[key])

   for shot in trainingSamples:
      appendToPrompt(prompt, shot, varGroup, question)
      prompt.append(shot['groundTruth'])
   appendToPrompt(prompt, testingSample, varGroup, question)
   # print(prompt)
   response = model.count_tokens(prompt)
   print(f"Prompt Token Count: {response.total_tokens}")
   for i in range(0,3):
      try:
         start = time.time()
         response = model.generate_content(prompt)
         end=time.time()
         print(f"Generate time taken: {end-start}")
         usage_metadata = response.usage_metadata
         print(f"Prompt Token Count: {usage_metadata.prompt_token_count}")
         return ' '.join(response.text.splitlines())
      except Exception as e:
         print(e)
         print("Retrying...")
         continue
   return "Error: Could not generate response"


def panelThought(trainingSamples, sample, varGroups, question):
   answers = []
   for varGroup in varGroups:
      response = generateResponse(trainingSamples, sample, varGroup, question)
      answers.append(response)
   return answers
   

def decisionTree(trainingSamples, testingSamples, varGroups, questions, filename):
   results = []
   count = 0
   for sample in testingSamples:
      result = []
      result.append(sample["currentDayFireData"]["current active fires"]["path"])
      result.append(sample["currentDayFireData"]["current active fires"]["fireCount"])
      result.append(sample["groundTruth"])

      if sample["currentDayFireData"]["current active fires"]["fireCount"] == 0:
         answers = panelThought(trainingSamples, sample, varGroups, questions["New Fire"])
         result.extend(answers)
         time.sleep(10)
      else:
         answers = panelThought(trainingSamples, sample, varGroups, questions["Spread"])
         time.sleep(10)
         smacount = 0
         simcount = 0
         larcount = 0
         for response in answers:
            if "Similar Active Fire" in response:
               simcount+=1
            elif "Smaller Active Fire" in response:
               smacount+=1
            elif "Larger Active Fire" in response:
               larcount+=1
         if smacount > simcount and smacount > larcount:
            answers = panelThought(trainingSamples, sample, varGroups, questions["No Fire"])
            time.sleep(10)
         result.extend(answers)
      results.append(result)
      count+=1
      if(count%50 == 0):
         df = np.array(copy.deepcopy(results))
         df = pd.DataFrame(df, columns= columns)   
         df.to_csv(filename + "(" + str(count) + ").csv", index=None)
   return results


vargroups = [["VIIRS band M11","VIIRS band I2", "VIIRS band I1", "NDVI", "EVI2", "energy release component", "current active fires"], ["total precipitation", "specific humidity", "Palmer drought severity index", "forecast total precipitation", "forecast specific humidity", "current active fires"], ["slope", "aspect", "elevation", "landcover class", "current active fires"], ["wind speed", "wind direction", "minimum temperature", "maximum temperature", "forecast wind speed", "forecast wind direction", "forecast temperature", "current active fires"]]
# vargroups = [["VIIRS band M11", "VIIRS band I2", "VIIRS band I1", "NDVI", "EVI2", "total precipitation", "wind speed", "wind direction", "minimum temperature", "maximum temperature", "energy release component", "specific humidity", "slope", "aspect", "elevation", "Palmer drought severity index", "landcover class", "forecast total precipitation", "forecast wind speed", "forecast wind direction", "forecast temperature", "forecast specific humidity", "current active fires"]]
# print(vargroups)

sampleBal = {
   "Larger Active Fire": 59,
   "Similar Active Fire": 59,
   "Smaller Active Fire": 59,
   "No Active Fire": 59,
   "New Active Fire": 59,
   "No New Active Fire": 59 
}
sampleCache = createSamples(varData, vargroups, "PNGs", sampleBal)
print(sampleBal)
print(len(sampleCache))
shotslocated = {
   "Larger Active Fire": 0,
   "Similar Active Fire": 0,
   "Smaller Active Fire": 0,
   "No Active Fire": 0,
   "New Active Fire": 0,
   "No New Active Fire": 0
}
testingSamples = []
# testingSamples = sampleCache
trainingSamples = []

for i in range(0,len(sampleCache)):
   loc = False
   for key in shotslocated.keys():
      if(sampleCache[i]['groundTruth'] == key and shotslocated[key] <9):
         print(sampleCache[i]['groundTruth'])
         trainingSamples.append(sampleCache[i])
         shotslocated[key]+=1
         loc = True
         break
   if(not loc):
      testingSamples.append(sampleCache[i])
print(shotslocated)
print(len(trainingSamples))
print(len(testingSamples))
   

columns = ["Path", "Current_Day_Fire_Count", "Ground_Truth"]
for varGroup in vargroups:
   columns.append("Group "+ ", ".join(varGroup) + "_GeminiResponse")
df = decisionTree(trainingSamples, testingSamples, vargroups, questions, "GeminiEval(Categorical)(DTP)(54Shots,4Groups,300Samples)(Balanced)")
df = np.array(df)
df = pd.DataFrame(df, columns= columns)   
df.to_csv("GeminiEval(Categorical)(DTP)(54Shots,4Groups,300Samples)(Balanced).csv", index=None)

# trainingSamples = []
# columns = ["Path", "Current_Day_Fire_Count", "Ground_Truth"]
# for varGroup in vargroups:
#    columns.append("Group "+ ", ".join(varGroup) + "_GeminiResponse")
# df = decisionTree(trainingSamples, testingSamples, vargroups, questions, "GeminiEval(Categorical)(DTP)(0Shots,4Groups,300Samples)(Balanced)")
# df = np.array(df)
# df = pd.DataFrame(df, columns= columns)   
# df.to_csv("GeminiEval(Categorical)(DTP)(0Shots,4Groups,300Samples)(Balanced).csv", index=None)