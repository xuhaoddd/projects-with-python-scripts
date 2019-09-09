#detect vibration in the camera and output the waveform
import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Camera
capture2 = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

#Camera
def CameraCapture():
    
    csvfile= open("data_.csv","a+",newline='')

    time_init = time.time()
    data_strage = str('')
    
    i=1
    i1=9
    i2=9
    
    #1-3000   300
    v = 3000 - 300 * i1
    #1-1000  100
    t_up_down = 1000 - 100 * i2
    max_amp = 0
    flag0=0
    
    
    x_list=[]
    time_list = []

    while True:
        
        
        #Time
        time_cycl = int((time.time() - time_init)*1000.0)

        #Image
        ret2,image2 = capture2.read()
        
       # out.write(image2)
        
       # cv2.imshow('frame',image2)
        
        gray=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        
        gray=np.float32(gray)
        
        dst=cv2.cornerHarris(gray,9,21,0.06)
        
        print(np.where(dst==dst.max()))
        
        x=np.where(dst==dst.max())[1][0]
        
        y=np.where(dst==dst.max())[0][0]
        
        if i==1:
            x0 = x
        
        i=i+1
        
        print(x-x0)
        
        x_list.append (x-x0)
        
        while ((x-x0) != 0) & (flag0==0) :
            t_start = time_cycl
            flag0 = 1
        
        if max_amp < np.abs(x-x0):
            max_amp = np.abs(x-x0)
        
        time_list.append (time_cycl)
        
        
        
        image2[dst==dst.max()]=[0,0,255]
        
        
        
        #gray[dst==dst.max()]=[0,0,255]
        
        #print(np.where(gray==[0,0,2
        
        cv2.circle(image2, (x,y) ,10, (0,0,255),-1)
        
        cv2.namedWindow("Capture2_corner_detection", cv2.WINDOW_NORMAL)
        
        cv2.imshow("Capture2_corner_detection", image2)
        
        out.write(image2)
        

        if ret2 == False:
            continue

        #Image Processing
########################################################
########            USER PROGRAM                ########
########################################################



        #Save Data
        data_strage += str(time_cycl) + "," + str(x-x0) + "\n"

########################################################


        #Make Window
        #cv2.namedWindow("Capture2", cv2.WINDOW_NORMAL)

        #cv2.imshow("Capture2", image2)
       #cv2.imwrite('image_test.png',image2)

        k = cv2.waitKey(1)
        if k == 27:
            t_end = time_cycl
            
            t_move= t_end - t_start
            '''
           # csvfile= open("data.csv","a")
            writer = csv.writer(csvfile)
           # writer.writerow(["v","t_up_down","amplitude","t_move"])
            writer.writerow([v,t_up_down,max_amp,t_move])
           # writer.writerows([[0,1,3],[1,2,3],[2,3,4]])
            csvfile.close()
            '''
                           
            with open('data_.txt', 'w') as fd:
                fd.write(data_strage)
            fd.close()
            break

    capture2.release()
    out.release()
    cv2.destroyAllWindows()
    
    # csvfile= open("data.csv","a")
    writer = csv.writer(csvfile)
           # writer.writerow(["v","t_up_down","amplitude","t_move"])
    writer.writerow([v,t_up_down,max_amp,t_move])
           # writer.writerows([[0,1,3],[1,2,3],[2,3,4]])
    csvfile.close()
    
    fig = plt.figure(1)
    
    plt.plot( time_list, x_list)
    
    plt.show()
    
    fig.savefig('hakei_%d_%d.png' % (v,t_up_down))
    
    
    

if __name__ == "__main__":
    CameraCapture()

