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
import cv2
import numpy as np
# 加?图像
img1 = cv2.imread('roi.jpg')
img2 = cv2.imread('opencv_logo.png')
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
# 取 roi 中与 mask 中不为?的值对应的像素的值?其他值为 0
# 注意??必?有 mask=mask 或者 mask=mask_inv, 其中的 mask= 不能忽略
img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
# 取 roi 中与 mask_inv 中不为?的值对应的像素的值?其他值为 0 。
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()