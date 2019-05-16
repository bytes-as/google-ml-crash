import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('1.jpg',0)
# cv2.namedWindow("Display window", cv2.WINDOW_NORMAL);		
# cv2.imshow("Display window", img)
# cv2.waitKey(0);

img1 = cv2.GaussianBlur(img, (5,5), 0)
plt.imshow(img1,cmap = 'gray')
plt.show()
cv2.waitKey(0);



# scale_percent = 50
# scaleArea = []
# input = img
# for _ in range(4):
# 	k = 100
# 	sigma = 1
# 	new_octave = []
# 	for k in range(5):
# 		sigma *= k
# 		new_octave.append(cv2.GaussianBlur(img, (5,5), sigma))
# 		sigma *= k
# 	scaleArea.append(new_octave)
# 	width = int(input.shape[1] * scale_percent / 100)
# 	height = int(input.shape[0] * scale_percent / 100)
# 	input = cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)

# for j in range(len(scaleArea)):
#     for i in range(len(scaleArea[j])):
#         plt.imshow(scaleArea[j][i],cmap = 'gray')
#         plt.title('Original[%d,%d]' % (j, i)), plt.xticks([]), plt.yticks([])
#         plt.show()
# cv2.waitKey(0);


# plt.subplot(1,1,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.show()