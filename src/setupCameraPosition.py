'''
Created on 21.01.2013
Here we position the camera to do measurement once the configuration is
well enough adjusted. 
@author: gerjer
'''

import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg    
from mpl_toolkits.mplot3d import Axes3D
import Image
import time
from colorConversion import *
from colorDifferences import *
from webcamToolbox import *

# Global variables
number_camera = 1
max_number_point_to_show = 5
max_number_frame_to_keep = 200
tabOscillographeDifferences = np.zeros((4,max_number_frame_to_keep))
    
def funShowWebcamStream(numSelectCamera):
    global tabOscillographeDifferences
    camera = initCamera(numSelectCamera)
    cv.NamedWindow("WindowWebcam")
    widthFrame = int(640)
    heightFrame = int(480)    
    
    # rectangle coordinates
    sub_rec1 = np.array([220,200,80,80])
    sub_rec2 = np.array([340,200,80,80])
    
    while True:
        frame = cv.QueryFrame(camera)
        print frame
        # conversion to Lab
        imFilter = cv.CreateImage((widthFrame,heightFrame), cv.IPL_DEPTH_8U,3)
        cv.CvtColor(frame, imFilter,cv.CV_BGR2Lab)
        imFilter = frame

        frame, dL, dLab, LabT, LabB, dY, XYZT, XYZB, dRGB = funDisplayLiveInformation(frame, widthFrame, heightFrame, sub_rec1, sub_rec2)
        
        # Here we do something to display data difference as an osilloscope
        tabOscillographeDifferences[:,0:-1] = tabOscillographeDifferences[:,1:]
        tabOscillographeDifferences[:,-1] = [dL, dLab, dY, dRGB]

        frame = funDisplayOscilliscope(frame, widthFrame, heightFrame, tabOscillographeDifferences)
        
        # --------------------------------------------
        cv.ShowImage("WindowWebcam",frame)
        if cv.WaitKey(10) == 113: # Leaves the live strean if q is pressed.
            print 'Il faut sortir maintenant, faut pas rester ici.'
            break    

    return 1

def main():
    global number_camera
    print 'The camera number is:',number_camera
    funShowWebcamStream(number_camera)
    return 1
      
# And now we start  
main()
print 'well done, you reach the biggest achievement of your day'
