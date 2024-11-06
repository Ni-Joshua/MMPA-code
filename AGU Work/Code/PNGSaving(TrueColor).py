import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import os
from PIL import Image
import google.generativeai as genai
import random
import copy

#This script reads the data from the directory where the data is downloaded from https://doi.org/10.5281/zenodo.8006177
#extracts the data from the 23 channels tif files and saves them as png files in the output directory
#it also creates 3 additional images for each fire event in the directory for testing purposes
#the first image is the original fire event, the second image is the fire event with half the number of cells, and the third image is the fire event with the number of cells doubled
#the fourth image is the fire event with the cells shifted by a random amount to simulate the fire event moving in a direction 
#The source directory structure is as follows: year/fires/ tif-files(that contain 23 features for each day)
#The output directory structure is as follows: year/fires/day/feature-images

channelnames = ["VIIRS band M11", "VIIRS band I2", "VIIRS band I1", "NDVI", "EVI2", "total precipitation", "wind speed", "wind direction", "minimum temperature", "maximum temperature", "energy release component", "specific humidity", "slope", "aspect", "elevation", "Palmer drought severity index", "landcover class", "forecast total precipitation", "forecast wind speed", "forecast wind direction", "forecast temperature", "forecast specific humidity", "active fires"]
sourceDirectory = "D:/WildfireSpreadTS" #update this to the directory where the data is downloaded from https://doi.org/10.5281/zenodo.8006177
# print(os.path.dirname(os.path.join(directory, filename)).split("/")[-1])
# maxvals = [-np.inf]*len(channelnames)
# minvals = [np.inf]*len(channelnames)
# random.seed(42)


# for year in os.listdir(sourceDirectory):
#     outputYearDirectory = os.path.join(outputDirectory, year)
#     sourceYearDirectory = os.path.join(sourceDirectory, year)
#     for folder in os.listdir(sourceYearDirectory):
#         sourceFireDirectory = os.path.join(sourceYearDirectory, folder)
#         outputFireDirectory = os.path.join(outputYearDirectory, folder)
#         print(sourceFireDirectory)
#         for filename in os.listdir(sourceFireDirectory):
#             with rasterio.open(os.path.join(sourceFireDirectory, filename), mode='r') as file:
#                 outputImagePath = os.path.join(outputFireDirectory, filename.split(".tif")[0] +"_lnglat_" +str(file.lnglat()))
#                 sourceImageArray = file.read()
#                 for i in range(0, len(sourceImageArray)):
#                     maxvals[i] = np.nanmax([maxvals[i], np.nanmax(sourceImageArray[i])])
#                     minvals[i] = np.nanmin([minvals[i], np.nanmin(sourceImageArray[i])])
#             file.close()
# np.savez("MinandMaxvals", minval = minvals, maxval = maxvals)

minmax = np.load("MinandMaxvals.npz")
maxvals = minmax["maxval"]
minvals = minmax["minval"]
print(maxvals)
print(minvals)

outputDirectory = "PNGs(TrueColor)(ActiveFirePatch)"
os.mkdir(outputDirectory)
for year in os.listdir(sourceDirectory):
    outputYearDirectory = os.path.join(outputDirectory, year)
    os.mkdir(outputYearDirectory)
    sourceYearDirectory = os.path.join(sourceDirectory, year)
    for folder in os.listdir(sourceYearDirectory):
        sourceFireDirectory = os.path.join(sourceYearDirectory, folder)
        outputFireDirectory = os.path.join(outputYearDirectory, folder)
        os.mkdir(outputFireDirectory)
        for filename in os.listdir(sourceFireDirectory):
            with rasterio.open(os.path.join(sourceFireDirectory, filename), mode='r') as file:
                outputImagePath = os.path.join(outputFireDirectory, filename.split(".tif")[0] +"_lnglat_" +str(file.lnglat()))
                os.mkdir(outputImagePath)
                sourceImageArray = file.read()
                print(sourceImageArray.shape)
                for i in range(0, len(sourceImageArray)):
                    if(channelnames[i] == "active fires"):
                        fireImage = np.nan_to_num(sourceImageArray[i], copy=True, nan=0.0)
                        minFireValue = np.nanmin(fireImage)
                        maxFireValue = np.nanmax(fireImage)
                        fireCellCount = np.count_nonzero(fireImage)
                        plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(minFireValue) +"," + str(maxFireValue) + ')' + '_fc('+str(fireCellCount) +').png', fireImage)
                        print(sourceImageArray.shape)
                    else:
                        plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(np.nanmin(sourceImageArray[i])) +"," + str(np.nanmax(sourceImageArray[i])) + ').png', sourceImageArray[i], vmin= minvals[i], vmax = maxvals[i])
#22141567, 9-24