import numpy as np
import os

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


""" Task 1 """

# Sample the images


# Create the triangle function


# Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)


""" Task 2 """

# Reconstruct the radiance using the calculated CRF





""" Task 3 """

# Perform both local and global tone-mapping

    

