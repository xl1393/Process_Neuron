import cv2
import numpy as np

img=cv2.imread('LossyTest.jp2')
'''
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

clahe=cv2.createCLAHE(clipLimit=10.0,tileGridSize=(10,10))
img_post=clahe.apply(img_gray)

img_result=cv2.cvtColor(img_post,cv2.COLOR_GRAY2BGR)

cv2.imwrite('clahe_lossytest.jp2',img_result)
'''

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
range=np.max(img_gray)-np.min(img_gray)
alpha=float(255)/range
beta=-np.min(img_gray) * alpha
#img2=cv2.convertTo(img_gray,alpha,beta)
img2=img_gray

img2=img_gray * alpha + beta

cv2.imwrite('ab_LossyTest.jp2',img2)
