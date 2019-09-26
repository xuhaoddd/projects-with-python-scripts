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

#mouse events cv2.setMouseCallback()
    #print mouse events
events=[i for i in dir(cv2) if 'EVENT'in i]
print(events)

    #double click paint circle
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
        #创建图像与窗口并将窗口与回调函数绑定
img = np.zeros((500, 500, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while (1):
    cv2.imshow('image', img)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cv2.destroyAllWindows()

        #paint rectangle and circle with mouse
drawing=False
mode=True
ix,iy=-1,-1
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                        #cv2.circle(img,(x,y),3,(0,0,255),-1)
                r=int(np.sqrt((x-ix)**2+(y-iy)**2))
                cv2.circle(img,(x,y),r,(0,0,255),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing==False
# if mode==True:
# cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
# else:
# cv2.circle(img,(x,y),5,(0,0,255),-1)

img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)
    if k==ord('m'):
        mode=not mode
    elif k==27:
        break
cv2.destroyAllWindows()


#slider 
def nothing(x):
    pass
#创建一个黑色图像
img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('R','image',0,255,nothing)

switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)

while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)
    if k == ord('q'):#按q键退出
        break

    b = cv2.getTrackbarPos('B','image')
    g = cv2.getTrackbarPos('G', 'image')
    r = cv2.getTrackbarPos('R', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:]=0
    else:
        img[:]=[b,g,r]
cv2.destroyAllWindows()










#mouse event  cv2.setMouseCallback()





    
    



 





















