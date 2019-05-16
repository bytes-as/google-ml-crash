import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('1.jpg',0)
edges = cv.Canny(img,100,50, True)
markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(markers, cmap='black')
plt.show()