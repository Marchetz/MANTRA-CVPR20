import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
import pdb
import os
import glob

colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.29, 0.57, 0.25)]
cmap_name = 'scene_list'
cm = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=4)

files = sorted(glob.glob('*.png'))
pdb.set_trace()
kernel = np.ones((5,5),np.uint8)
pdb.set_trace()
for f in files:

    image = cv2.imread(f, 0)
    image_copy = image.copy()
    image_copy[np.where(image_copy == 2)] = 0
    image_copy[np.where(image_copy == 3)] = 0
    image_copy[np.where(image_copy == 4)] = 0
    closing = cv2.morphologyEx(image_copy, cv2.MORPH_CLOSE, kernel)
    image[np.where(closing == 1)] = 1
    cv2.imwrite('closing/' + f[:-4] + '.png', image)

    #to visualize with colors
    #plt.imshow(image,cm)
    #plt.savefig('png_closing/color/' + f[:-4] + '_5.png')






