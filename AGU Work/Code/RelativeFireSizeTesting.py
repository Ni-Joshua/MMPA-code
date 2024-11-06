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

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  start = time.time()
  file = genai.upload_file(path, mime_type=mime_type)
  end = time.time()
  print(f"Uploaded file '{file.display_name}' as: {file.uri} in {end-start} seconds")
  return file

def buildFireData(currentDayDirectory):
   fireData = {}
   fileList = os.listdir(currentDayDirectory)
   for file in fileList:
      if("shrunk" in file or "expand" in file or "shift" in file):
         continue
      varName = file.split("_")[0]
      if(varName == 'active fires'):
         varName = 'current active fires'
      fireData[varName] = {}
      fireData[varName]["minVal"] = file.split("range(")[-1].split(")")[0].split(',')[0]
      fireData[varName]["maxVal"] = file.split("range(")[-1].split(")")[0].split(',')[1]
      fireData[varName]['path'] = os.path.join(currentDayDirectory, file)
      if "current active fires" in varName:
         fireCount = file.split("_fc(")[-1].split(")")[0]
         fireData[varName]["fireCount"] = int(fireCount)   
   return fireData

def createSample(currentDayDirectory, nextDayDirectory):
   currentDayFireData = buildFireData(currentDayDirectory)
   nextDayFireData = buildFireData(nextDayDirectory)
   sample  = { "currentDayFireData": currentDayFireData,
                "nextDayFireData": nextDayFireData,}
   key = "prompt"
   prompt = createVarGroupPrompt(currentDayFireData)
   sample[key] = prompt
   return sample

def createVarGroupPrompt(fireData):
   prompt = []
   prompt.append("input: You are a wildfire expert. Here is an image of the current active fires in an area. Each pixel represents a 375m by 375m area. A purple pixel represents no active fire and any other colored pixel represents an active fire.")
   prompt.append(upload_to_gemini(fireData["current active fires"]["path"], mime_type="image/png"))
#1    prompt.append("What is the size of this fire? Answer with the following options: Small, Large, Medium, None. Please provide reasoning for your response and an estimate for the number of active fire pixels in the image. output:")
#2    prompt.append("What is the size of this fire, found from the total number of burn pixels (not necessarily together)? Answer with the following options: Small, Large, Medium, None. Please provide reasoning for your response and an estimate for the number of active fire pixels in the image. output:")
   prompt.append("What is the overall size of the fire(s)? Answer with the following options: Small, Large, Medium, None. Please provide reasoning for your response and an estimate for the number of active fire pixels in the image. output:")
   return prompt

def createFixedSamples(directories):
   samples = []
   for folder in directories.keys():
      currentDayDirectory = folder
      followingDayDirectory = directories[folder]
      sample = createSample(currentDayDirectory, followingDayDirectory)
      samples.append(sample)
   return samples

def generateResponse(sample):
    prompt = sample['prompt']
    response = model.count_tokens(prompt)
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

# def prompting(directories):
#     df = []
#     for key in directories.keys():

def test(filename, columns, samples):
   results = []
   count = 0
   for sample in samples:
      result = []
      result.append(sample["currentDayFireData"]["current active fires"]["path"])
      result.append(sample["currentDayFireData"]["current active fires"]["fireCount"])
      result.append(generateResponse(sample))
      results.append(result)
      count+=1
      if(count%50 == 0):
         df = np.array(copy.deepcopy(results))
         df = pd.DataFrame(df, columns= columns)   
         df.to_csv(filename + "(" + str(count) + ").csv", index=None)
   return results


def createSamples(rootDir, samplecount, havezero):
   samples = []
   paths = []
   i = 0
   while(i < samplecount):
      yearFolder = os.listdir(rootDir)[random.randint(0, len(os.listdir(rootDir))-1)]
      yearDirectory = os.path.join(rootDir, yearFolder)
      fireFolder = os.listdir(yearDirectory)[random.randint(0, len(os.listdir(yearDirectory))-1)]
      fireDirectory = os.path.join(yearDirectory, fireFolder)
      dayid = random.randint(0, len(os.listdir(fireDirectory))-2)
      currentDayFolder = os.listdir(fireDirectory)[dayid]
      nextDayFolder = os.listdir(fireDirectory)[dayid+1]
      currentDayDirectory = os.path.join(fireDirectory, currentDayFolder)
      followingDayDirectory = os.path.join(fireDirectory, nextDayFolder)
      if(followingDayDirectory in paths):
         continue
      else:
         paths.append(followingDayDirectory)
      sample = createSample(currentDayDirectory, followingDayDirectory)
      if(sample["currentDayFireData"]["current active fires"]["fireCount"] == 0):
         continue
      i+=1
      samples.append(sample)
   return samples

# directories = {
#    #Testing
#    "PNGs/2021/fire_25295951/2021-07-12_lnglat_(-106.75552110157054, 45.23744401775735)": "PNGs/2021/fire_25295951/2021-07-13_lnglat_(-106.75552110157054, 45.23744401775735)",
#    "PNGs/2021/fire_25295951/2021-07-13_lnglat_(-106.75552110157054, 45.23744401775735)": "PNGs/2021/fire_25295951/2021-07-14_lnglat_(-106.75552110157054, 45.23744401775735)",
#    "PNGs/2021/fire_25295023/2021-07-23_lnglat_(-117.59959061565829, 45.87009402770428)": "PNGs/2021/fire_25295023/2021-07-24_lnglat_(-117.59959061565829, 45.87009402770428)",
#    "PNGs/2021/fire_25295951/2021-07-20_lnglat_(-106.75552110157054, 45.23744401775735)": "PNGs/2021/fire_25295951/2021-07-21_lnglat_(-106.75552110157054, 45.23744401775735)",
#    "PNGs/2021/fire_25295951/2021-07-21_lnglat_(-106.75552110157054, 45.23744401775735)" : "PNGs/2021/fire_25295951/2021-07-22_lnglat_(-106.75552110157054, 45.23744401775735)",
#    "PNGs/2021/fire_25295951/2021-07-22_lnglat_(-106.75552110157054, 45.23744401775735)": "PNGs/2021/fire_25295951/2021-07-23_lnglat_(-106.75552110157054, 45.23744401775735)",
#    "PNGs/2021/fire_25295856/2021-07-25_lnglat_(-107.57868690446206, 47.57230644965683)":"PNGs/2021/fire_25295856/2021-07-26_lnglat_(-107.57868690446206, 47.57230644965683)"
# }
# print(directories)

# samples = createFixedSamples(directories)

# samples = createSamples("PNGs", 500, true)
# # samples = createFixedSamples({"PNGs/2021/fire_25294691/2021-08-29_lnglat_(-123.13533697890595, 41.04095686845025)": "PNGs/2021/fire_25294691/2021-08-30_lnglat_(-123.13533697890595, 41.04095686845025)"})
# columns = ["Path", "Current_Day_Fire_Count","Response"]
# df = []
# df = test("GeminiEval(KnowledgeTesting)2", columns, samples)
# df = np.array(df)
# df = pd.DataFrame(df, columns= columns)   
# df.to_csv("GeminiEvalGeminiEval(KnowledgeTesting)2.csv", index=None)

samples = createSamples("PNGs", 20, False)
print(samples)
columns = ["Path", "Current_Day_Fire_Count","Response"]
df = []
for i in range(0, 20):
   df.extend(test("GeminiEval(KnowledgeTesting)(Consistency)" + str(i), columns, samples))

df = np.array(df)
df = pd.DataFrame(df, columns= columns)   
df.to_csv("GeminiEvalGeminiEval(KnowledgeTesting)(Consistency).csv", index=None)