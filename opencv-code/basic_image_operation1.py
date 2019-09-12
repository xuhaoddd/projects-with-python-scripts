import cv2
import numpy as np
import matplotlib.pyplot as plt

#obtain and revise pixel value,image shape,ROI,image channel
img=cv2.imread('E:/test image/lena.png')
img.shape
px=img[100,100]
blue=img[100,100,0]
img[100,100]=[255,255,255]
img.item(100,100,0) #return scalar
img.itemset((100,100,0),200)
img.size
img.dtype
region=img[280:340,330:390]
b,g,r=cv2.split(img)
img1=cv2.merge([r,g,b])# cv2.merge([])
plt.imshow(img1),plt.title('rgb_lena')
plt.show()
b=img[:,:,0]
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
#conclude: img.item(), img.itemset(), cv2.split().cv2.merge([]),
# img.dtype/size/shape,cv2.copyMakeBorder

x = np.uint8([250])
y = np.uint8([10])
print(x+y)
print(cv2.add(x,y))

dst=cv2.addWeighted(img,0.7,img1,0.3,0) #dst = α · img + β · img1 + γ
cv2.imshow('blend',dst)
cv2.waitKey(0)
cv2.destroyAllWindow()

# threshold https://blog.csdn.net/on2way/article/details/46812121
