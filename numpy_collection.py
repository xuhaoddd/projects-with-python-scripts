# -*- coding: utf-8 -*-
import numpy as np

#1-dimension array
    #create ndarray
array1=np.array([1,2,3])
array2=np.ones(3)
array3=np.zeros(3)
array4=np.random.random(3)
    #calculation of array
array1 * 1.6
array1 + array2

    #index of array
array1[0]
array1[0:2]
array1[1:]

    #basic methematic function of array
array1.min()
array1.sum()

######################################################################
#multi-denmension array
array5=np.array([[1,2],[3,4],[5,6]])
array6=np.ones([3,3])
    #broadcast operation
array6 + array1

    #dot (no difference between column and row in 1-dimension array)
array6.dot(array1)
array1.dot(array6)

    #index
array5[0,1]
array5[1:3]
array5[1:3,1]

    #basic methematic function(pay attention to axis)
array5.min(axis=0)
array5.min(axis=1)

    #transpose and reshape
array5.T
array5.reshape(2,3)

    #more than 2-dim (3-dim correspond to the left-hand coordinate)
array7=np.random.random([4,3,2])


#complex calculation function
    #Mean square error
#error = (1/n) * np.sum(np.square(predictions - labels))
    
    
    
#applications
    #table
#pandas.DataFrame and cvs
    
    
    
    #audio
#1-dimension array
    
    #image
#Binary image(2-dimension array) and color image(3-dimension array)