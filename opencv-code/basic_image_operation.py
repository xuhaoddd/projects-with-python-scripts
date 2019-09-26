# -*- coding: utf-8 -*-
import cv2 
import numpy as np
import matplotlib.pyplot as plt

#read, show, write image
    #read the image with two ways,IMREAD_GRAYSCALE and IMREAD_COLOR
    #path shold be "/"
img=cv2.imread("E:/test images/lena.png",cv2.IMREAD_GRAYSCALE)

    #cv2.WINDOW_NORMAL(adjust the size of window) / cv2.WINDOW_AUTOSIZE
cv2.namedWindow("image0", cv2.WINDOW_NORMAL)
cv2.imshow('image0',img)

    #waitkey(0) wait the keyboard input forever
cv2.waitKey(0)

    #when destroying the window, the window name is needed
cv2.destroyWindow('image0')

    #write image
cv2.imwrite("E:/test images/lena_gray.jpg",img)

    #example
img1 = cv2.imread('E:/test images/lena.png',cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('image1',cv2.WINDOW_AUTOSIZE)
cv2.imshow('image1', img1)
key = cv2.waitKey(0)
#ord() is Built-in function in python, it returns the ASCII 
if key == ord('s'):
    cv2.imwrite('E:/test images/lena_gray.jpg',img1)
    cv2.destroyWindow('image1')
elif key == 27:
    cv2.destroyWindow('image1')


    #practice

img = cv2.imread('E:/test images/lena.png')
#pay attention to split() and merge()
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])

plt.subplot(121);plt.xticks([]); plt.yticks([]);plt.imshow(img) # expects distorted color
plt.subplot(122);plt.xticks([]); plt.yticks([]);plt.imshow(img2) # expect true color
plt.show()

cv2.imshow('bgr image',img) # expects true color
cv2.imshow('rgb image',img2) # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()


#vedio processing

    #open camera or video file

    #cap.isOpened() can examine if the initialization is successful
    #cap=cv2.VideoCapture('xxx.avi')
cap=cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    #ret (True or False) represent if the frame has been read successfully
    ret,frame=cap.read()
    if ret==True:
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyWindow('frame')


#plot in the gui
    #cv2.line() cv2.circle() cv2.rectangle() cv2.ellipse() cv2.putText()
    #plot line
img=np.zeros((512,512,3), np.uint8)
        ## Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)

    #plot rectangle
        ## Draw a diagonal blue line with thickness of 5 px
        #give the upper left corner and lower right corner coordinates of rectangle
        #origin is in the the upper left corner
cv2.rectangle(img,(100,0),(510,128),(0,255,0),3)

    #plot circle
cv2.circle(img,(447,63), 63, (0,0,255), -1)

    #plot ellipse
cv2.ellipse(img, (256, 100), (100, 100), 60, 0, 300, (0, 0, 255), -1, cv2.LINE_AA)


pts = np.array([[100, 5],  [300, 100], [300, 200], [100, 300]], np.int32)
    #plot polylines
        # 顶点个数：4，矩阵变成4*1*2维
        #pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 0, 255), 10)

    #input text
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)

cv2.imshow('black with line',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#mouse event  cv2.setMouseCallback()





    
    



 





















