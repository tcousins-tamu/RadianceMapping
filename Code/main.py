import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from gsolve import gsolve
from utils import *
# Based on code by James Tompkin
#
# reads in a directory and parses out the exposure values
# files should be named like: "xxx_yyy.jpg" where
# xxx / yyy is the exposure in seconds. 
def ParseFiles(calibSetName, dir):
    imageNames = os.listdir(os.path.join(dir, calibSetName))
    
    filePaths = []
    exposures = []
    
    for imageName in imageNames:
        exposure = imageName.split('.')[0].split('_')
        exposures.append(int(exposure[0]) / int(exposure[1]))
        filePaths.append(os.path.join(dir, calibSetName, imageName))
    
    # sort by exposure
    sortedIndices = np.argsort(exposures)[::-1]
    filePaths = [filePaths[i] for i in sortedIndices]
    exposures = [exposures[i] for i in sortedIndices]
    
    return filePaths, exposures

# Setting up the input output paths and the parameters
inputDir = '../Images/'
outputDir = '../Results/'

_lambda = 50

calibSetName = 'Chapel'

# Parsing the input images to get the file names and corresponding exposure
# values
filePaths, exposures = ParseFiles(calibSetName, inputDir)

#going to start with one file
index = 0 
""" Task 1 """
# Choosing the pixel locations
images = []
for file in (filePaths):
    images.append(cv2.cvtColor(plt.imread(file), cv2.COLOR_BGR2GRAY)*255) #gets it in range from 0-255
images = np.asarray(images, dtype = np.uint8)
print("Image max: ", np.max(images), "images Size: ", images.shape)
pixels = np.random.randint(0, (images[0].shape[0], images[0].shape[1]), size = (int(5*256/(len(filePaths)-1)), 2))

# Sample the images
# Create the triangle function
# Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
Z = np.zeros((len(pixels), len(images)), dtype = int) #Pixel Values of locations in j
for p in range(len(pixels)):
    for i in range(len(images)):
        Z[p, i] = images[i][pixels[p][0]][pixels[p][1]]

Zmax = np.max(Z)
Zmin = np.min(Z)

# Calculate the log shutter speed
B = np.log(exposures) #Log Shutter Speed for an image, j
W = Weights(Zmax, Zmin) #Weight for pixel value Z (according to EQ 4, debevec)

g, le = gsolve(Z, B, _lambda, W)

#figure out how to plot this
print("TASK ONE DONE")
""" Task 2 """

# Reconstruct the radiance using the calculated CRF





""" Task 3 """

# Perform both local and global tone-mapping

    

