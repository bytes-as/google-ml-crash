
# coding: utf-8

# In[213]:


import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


# In[224]:


img = cv2.imread('1.jpg',0)
plt.imshow(img,cmap = 'gray')
plt.show()


# In[225]:


scaleArea = []
input = img
for _ in range(4):
    sigma = (_+1)/math.sqrt(2)
    new_octave = []
    to_be_blurred = input
#     new_octave.append(to_be_blurred)
    for _ in range(5):
        blurred = cv2.GaussianBlur(to_be_blurred, (3,3), sigma)
        new_octave.append(blurred)
        to_be_blurred = blurred
        sigma *= math.sqrt(2)
    scaleArea.append(new_octave)
    width = int(input.shape[1] * 50 / 100)
    height = int(input.shape[0] * 50 / 100)
    input = cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)


# In[226]:


print("scaleArea.shape = (%d,%d)" %(len(scaleArea), len(scaleArea[0])))


# In[227]:


for j in range(len(scaleArea)):
    for i in range(len(scaleArea[j])):
        plt.imshow(scaleArea[j][i],cmap = 'gray')
        plt.title('Original[%d,%d]' % (j, i)), plt.xticks([]), plt.yticks([])
        plt.show()


# In[228]:


log = []
for i in range(len(scaleArea)):
    new_log = []
    for j in range(len(scaleArea[i]) - 1):
        new_log.append(scaleArea[i][j+1] - scaleArea[i][j])
    log.append(new_log)


# In[229]:


print("log.shape = (%d,%d)" % (len(log),len(log[0]))) 


# In[230]:


for i in range(len(log)):
    for j in range(len(log[i])):
        plt.imshow(log[i][j],cmap = 'gray')
        plt.title('log[%d,%d]' % (i, j)), plt.xticks([]), plt.yticks([])
        plt.show()


# In[257]:


# log = [[np.zeros((3,3)),np.array([[1,1,1],[0,-1,1],[1,1,1]]),np.zeros((3,3))]]
# above log is just for testing the fucntion to find the local minima and  maxima

# this link is by far most helpful (may be) in finding sub-pixel
# https://stackoverflow.com/questions/9532415/sift-taylor-expansion-working-out-subpixel-locations?rq=1

keypoints = []
for i in range(len(log)):
    keypoint = []
    for j in range(1, len(log[i]) - 1):
        new_img = np.zeros(log[i][j].shape)
        for y in range(1, int(log[i][j].shape[0]) - 1):
            for x in range(1, int(log[i][j].shape[1]) - 1):
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(log[i][j][y-1:y+1,x-1:x+1], mask=None)
                if minVal == log[i][j][y,x]:
                    min_minVal, min_maxVal, min_minLoc, min_maxLoc = cv2.minMaxLoc(log[i][j-1][y-1:y+1,x-1:x+1], mask=None)
                    max_minVal, max_maxVal, max_minLoc, max_maxLoc = cv2.minMaxLoc(log[i][j+1][y-1:y+1,x-1:x+1], mask=None)
                    if minVal < min_minVal and minVal < max_minVal:
                        new_img[y,x] = 255
                elif maxVal == log[i][j][y,x]:
                    min_minVal, min_maxVal, min_minLoc, min_maxLoc = cv2.minMaxLoc(log[i][j-1][y-1:y+1,x-1:x+1], mask=None)
                    max_minVal, max_maxVal, max_minLoc, max_maxLoc = cv2.minMaxLoc(log[i][j+1][y-1:y+1,x-1:x+1], mask=None)
                    if maxVal > min_maxVal and maxVal > max_maxVal:
                        new_img[y,x] = 255
                else:
                    new_img[y,x] = 0
        keypoint.append(new_img)
    keypoints.append(keypoint)


# In[232]:


print("keypoints.shape = (%d,%d)" % (len(keypoints), len(keypoints[0])))


# In[233]:


for i in range(len(keypoints)):
    for j in range(len(keypoints[i])):
        plt.imshow(keypoints[i][j],cmap = 'gray')
        plt.title('log[%d,%d]' % (i, j)), plt.xticks([]), plt.yticks([])
        plt.show()


# In[285]:


def deriv_3D(dog, octv, intvl, r, c):
    dx = (dog[octv][intvl][r, c+1] - dog[octv][intvl][r, c-1])/2
    dy = (dog[octv][intvl][r+1, c] - dog[octv][intvl][r-1, c])/2
    ds = (dog[octv][intvl+1][r, c] - dog[octv][intvl-1][r, c])/2
    return np.array([dx, dy, ds])


# In[286]:


def hessian_3D(dog, octv, intvl, r, c):
    v = dog[octv][intvl][r,c]
    dxx = dog[octv][intvl][r,c+1] + dog[octv][intvl][r,c-1] - 2*v
    dyy = dog[octv][intvl][r+1,c] + dog[octv][intvl][r-1,c] - 2*v
    dss = dog[octv][intvl+1][r,c] + dog[octv][intvl-1][r,c] - 2*v
    dxy = (dog[octv][intvl][r+1,c+1] + dog[octv][intvl][r-1,c-1] -
           dog[octv][intvl][r-1,c+1] - dog[octv][intvl][r+1,c-1] )/4
    dxs = (dog[octv][intvl+1][r,c+1] + dog[octv][intvl-1][r,c-1] -
           dog[octv][intvl-1][r,c+1] - dog[octv][intvl+1][r,c-1] )/4
    dys = (dog[octv][intvl+1][r+1,c] + dog[octv][intvl-1][r-1,c] -
           dog[octv][intvl+1][r-1,c] - dog[octv][intvl-1][r+1,c] )/4
    return np.array([[dxx, dxy, dxs],
                        [dxy, dyy, dys],
                        [dxs, dys, dss]])


# In[380]:


def interp_step(dog, octv, intvl, r, c):
    dD = deriv_3D(dog, octv, intvl, r, c)
    H = hessian_3D(dog, octv, intvl, r, c)
    H_inv = np.zeros(H.T.shape)
    cv2.invert(H, H_inv, cv2.DECOMP_SVD)
#     print("H_inv:")
#     print(H_inv)
#     print(H_inv.shape)
#     print("dD")
#     print(dD)
#     print(dD.shape)
    gm = cv2.gemm(H_inv, dD, -1, None, 0)
    return gm


# In[417]:


def interp_contr(dog, octv, intvl, r, c):#, xi, xr, xc):
    [xi, xr, xc] = interp_step(dog, octv, intvl, r, c)
    dD = np.array([deriv_3D(dog, octv, intvl, r, c)])
    X = np.array([xc, xr, xi])
    m = np.dot(dD, X)
    contra = dog[octv][intvl][r,c] + m[0] * 2
    return contra[0]


# In[418]:


# print(interp_step(log, 0, 0, 100, 100))
contra = []
for r in range(1, log[0][0].shape[0]-1):
    print(r)
    new_contra = []
    for c in range(1, log[0][0].shape[1]-1):
        new_contra.append(interp_contr(log, 0, 0, r, c))
    contra.append(np.array(new_contra))


# In[422]:


contra = np.array(contra)
print(contra)
print(contra.shape)
x = np.linspace(1, log[0][0].shape[1]-1, log[0][0].shape[1]-2)
print(x.shape)
y = np.linspace(1, log[0][0].shape[0]-1, log[0][0].shape[0]-2)
print(y.shape)


# In[427]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print(y)
print(x)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, contra)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
######################################
