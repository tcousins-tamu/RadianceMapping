import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from gsolve import gsolve
from utils import *
from skimage.filters import gaussian
import matplotlib.colors as clrs
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
gamma = .2
std_dev = 1

calibSetName = 'Chapel'
#calibSetName = 'Office'

# Parsing the input images to get the file names and corresponding exposure
# values
filePaths, exposures = ParseFiles(calibSetName, inputDir)

#going to start with one file
index = 0
""" Task 1 """
# Choosing the pixel locations
images = []
for file in (filePaths):
    images.append(plt.imread(file)*255) #gets it in range from 0-255
images = np.asarray(images, dtype = np.uint8)
pixels = np.random.randint(0, (images[0].shape[0], images[0].shape[1]), size = (int(5*256/(len(filePaths)-1)), 2))
# Sample the images
# Create the triangle function
# Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
colors = ['red', 'green', 'blue']
g = np.zeros((images.shape[-1], 256), dtype = float) #Pixel Values
for channel in range(images.shape[-1]):
    Z = np.zeros((len(pixels), len(images)), dtype = int) #Pixel Values of locations in j
    for p in range(len(pixels)):
        for i in range(len(images)):
            Z[p, i] = images[i][pixels[p][0]][pixels[p][1]][channel]
            
    Zmax = 255
    Zmin = 0 

    # Calculate the log shutter speed
    B = np.log(exposures) #Log Shutter Speed for an image, j
    W = Weights(Zmax, Zmin) #Weight for pixel value Z (according to EQ 4, debevec)
    g[channel], le = gsolve(Z, B, _lambda, W)
    plt.plot(g[channel], range(256), color = colors[channel], linewidth = 2)
plt.title("CRF")
plt.show()

""" Task 2 """
#Reconstructing the radiance from the CRF and the input images
radiance = np.zeros(images[0].shape)
weightsSum  = np.zeros(images[0].shape)
for chan in range(images.shape[-1]):
    for img in range(len(filePaths)):
        holder = (W[images[img][:,:,chan]])
        radiance[:, :, chan] += (holder* (g[chan, images[img, :, :, chan]] - B[img])).reshape(images[0].shape[:2])
        weightsSum[:, :, chan] += holder
    
weightsSum[weightsSum == 0] = 1e-100
for chan in range(images.shape[-1]):
    radiance[:, :, chan]/=weightsSum[:,:, chan]

radiance = np.exp(radiance)

rad1D = radiance[:, :, 0] + radiance[:, :, 1] + radiance[:, :,2]
plt.imshow(rad1D, cmap="jet", norm=clrs.LogNorm(vmin=radiance.min(), vmax=radiance.max()))
cbar = plt.colorbar()
plt.title('Radiance')
plt.show()

""" Task 3 """
# Perform both local and global tone-mapping
#Global Tone Mapping
globTone = (radiance/np.max(radiance))**(.1)
plt.imshow(globTone)
plt.title('Global Tone Mapping')
plt.show()



#Local Tone Mapping
intensity = np.zeros(radiance.shape[-1])
chrominance = np.zeros(radiance.shape)
intensity = np.mean(radiance, axis = -1)
intensity = np.abs(intensity)
for chan in range(images.shape[-1]):
    chrominance[:,:,chan] = radiance[:,:,chan]/intensity #[:,:,chan]

lInt = np.log2(intensity) #the average of the color channels would be three digits, I think this refers to chrominance
lIntFilt = gaussian(lInt, std_dev)

detail = lInt - lIntFilt
lIntFilt = (lIntFilt - np.max(lIntFilt))*(4/(np.max(lIntFilt) - np.min(lIntFilt))) #4 is an arbitrary scalar selected from the doc

lItensity = np.power(2, lIntFilt+detail)
out = np.zeros(radiance.shape)
for chan in range(images.shape[-1]):
    out[:, :, chan] = chrominance[: ,:, chan]*lItensity
out = out**(.5)
plt.imshow(out)
plt.title('Local Tone Mapping')
plt.show()