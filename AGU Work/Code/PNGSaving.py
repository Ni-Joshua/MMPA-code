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
# print(os.path.dirname(os.path.join(directory, filename)).split("/")[-1])
random.seed(42)
sourceDirectory = "D:/WildfireSpreadTS" #update this to the directory where the data is downloaded from https://doi.org/10.5281/zenodo.8006177
outputDirectory = "PNGs"
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
                # print(file.read().shape)
                # print(file.tags())
                # print(file.meta)
                # print(file.lnglat())
                sourceImageArray = file.read()
                for i in range(0, len(sourceImageArray)):
                    if(i == len(sourceImageArray)-1):
                        fireImage = np.nan_to_num(sourceImageArray[i], copy=True, nan=0.0)
                        minFireValue = np.min(fireImage)
                        maxFireValue = np.max(fireImage)
                        fireCellCount = np.count_nonzero(fireImage)


                        edges = set()
                        expandedCells = set()
                        fireCells = []
                        for indx in range(0, len(fireImage)):
                            for indy in range(0, len(fireImage[indx])):
                                if(fireImage[indx][indy]!=0):
                                    fireCells.append((indx,indy))
                                    if(indx>0):
                                        if (fireImage[indx-1][indy]==0):
                                            edges.add((indx,indy))
                                            expandedCells.add((indx-1,indy))
                                    if(indx<len(fireImage)-1):
                                        if(fireImage[indx+1][indy]==0):
                                            edges.add((indx,indy))
                                            expandedCells.add((indx+1,indy))
                                    if(indy>0):
                                        if(fireImage[indx][indy-1]==0):
                                            edges.add((indx,indy))
                                            expandedCells.add((indx, indy-1))
                                    if(indy<len(fireImage[indx])-1):
                                        if(fireImage[indx][indy+1]==0):
                                            edges.add((indx,indy))
                                            expandedCells.add((indx,indy+1))
                                            
                        expandedFireImage = copy.deepcopy(fireImage)
                        timevals = np.delete(np.unique(fireImage), np.where(np.unique(fireImage) == 0))
                        for (x1,y1) in expandedCells:
                            expandedFireImage[x1][y1] = timevals[random.randint(0, len(timevals)-1)]
                        expandedFireImage = np.array(expandedFireImage)
                        
                        currentcount = np.count_nonzero(fireImage)
                        shrunkenFireImage = copy.deepcopy(fireImage)
                        edges = np.array(edges)
                        while(currentcount> int(fireCellCount/2) and edges.size>0):
                            toBeRemovedCell = random.randint(0, len(edges)-1)
                            shrunkenFireImage[edges[toBeRemovedCell][0]][edges[toBeRemovedCell][1]] = 0
                            edges = np.delete(edges, toBeRemovedCell, axis=0)
                            currentcount-=1
                        shrunkenFireImage = np.array(shrunkenFireImage)

                        shiftedFireImage = np.zeros_like(fireImage)
                        fireCells=np.array(fireCells)
                        if(len(fireCells)>0):
                            xdiff = random.randint(-1*np.min(fireCells[:,0]), len(fireImage)-1-np.max(fireCells[:,0]))
                            ydiff = random.randint(-1*np.min(fireCells[:,1]), len(fireImage[0])-1-np.max(fireCells[:,1]))
                            while(xdiff == 0 and ydiff ==0):
                                xdiff = random.randint(-1*np.min(fireCells[:,0]), len(fireImage)-1-np.max(fireCells[:,0]))
                                ydiff = random.randint(-1*np.min(fireCells[:,1]), len(fireImage[0])-1-np.max(fireCells[:,1]))
                            # print(burncells, xdiff,ydiff, x.shape)
                            for (x1,y1) in fireCells:
                                shiftedFireImage[x1+xdiff][y1+ydiff] = fireImage[x1][y1]
                            plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(minFireValue) +"," + str(maxFireValue) + ')' + '_fc('+str(np.count_nonzero(shiftedFireImage)) +')(shift)_(' +str(xdiff) + ", "+ str(ydiff)+").png", shiftedFireImage)
                        else:
                            plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(minFireValue) +"," + str(maxFireValue) + ')' + '_fc('+str(np.count_nonzero(shiftedFireImage)) +')(shift)_(0,0).png', shiftedFireImage)
                        plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(minFireValue) +"," + str(maxFireValue) + ')' + '_fc('+str(np.count_nonzero(shrunkenFireImage)) +')(shrunk).png', shrunkenFireImage)
                        plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(minFireValue) +"," + str(maxFireValue) + ')' + '_fc('+str(np.count_nonzero(expandedFireImage)) +')(expanded).png', expandedFireImage)
                        plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(minFireValue) +"," + str(maxFireValue) + ')' + '_fc('+str(fireCellCount) +').png', fireImage)

                    else:
                        plt.imsave(outputImagePath + "/"+channelnames[i] + "_range(" + str(np.min(sourceImageArray[i])) +"," + str(np.max(sourceImageArray[i])) + ').png', sourceImageArray[i])
#22141567, 9-24