from scipy.misc import imread
from scipy.misc import imsave
import os,sys
import numpy as np
from sklearn import preprocessing
from PIL import Image

img=imread('3channel.png')
print type(img)
print img.shape

for i in range(3):
    ss=0
    for x in range(2000):
        for y in range(2000):
            ss=img[x,y,i]+ss
    print np.sum(img[:,:,i])/float(2000**2)
    mu=ss/float(2000*2000)
    print mu
    s=0
    for x in range(2000):
        for y in range(2000):
            s=s+(img[x,y,i]-mu)**2
    #sigma=np.sum((img[:,:,i]-mu)**2)/float(2000**2)
    sigma=np.sqrt(s/float(2000**2))
    print sigma
    for x in range(2000):
        for y in range(2000):
           # print img[x,y,i]
            img[x,y,i]=(img[x,y,i]-mu)/float(sigma)
            #print img[x,y,i]
imsave('test_whitened.png',img)