import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from gsolve import gsolve
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
    images.append(cv2.cvtColor(plt.imread(file), cv2.COLOR_BGR2GRAY))
images = np.asarray(images)
pixels = np.random.randint(0, (images[0].shape[0], images[0].shape[1]), size = (int(5*256/(len(filePaths)-1)), 2))

# Sample the images
# Create the triangle function
# Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
Z = [] #Pixel Values of locations in j
B = [] #Log Shutter Speed for an image, j
W = [] #Weight for pixel value Z (according to EQ 4, debevec)

gsolve(Z, B, _lambda, W)
# g = np.zeros((256, 1))
# l = np.zeros((256, len(images)))
# w = np.zeros((256, len(images)))
# z = np.zeros((len(images), 1))
# for rad in range(256):
#     indices = np.where(images[pixels]==rad)
#     z[:, 0] = images[pixels][indices]
#     w[rad, :] = exposures[indices[0]]
#     l[rad, :] = np.log(exposures[indices[0]])
#     g[rad] = np.sum(z[:, 0]/w[rad, :])/np.sum(1/w[rad, :])
# crf = np.exp(g)


""" Task 2 """

# Reconstruct the radiance using the calculated CRF





""" Task 3 """

# Perform both local and global tone-mapping

    

