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
max_number_point_to_show = 5
max_number_frame_to_keep = 200
tab_oscillographe_differences = np.zeros((4,max_number_frame_to_keep))
    
def function_show_webcam_stream(type_camera_number = 0):
    ''' The function display a webcam stream and overlay some information 
    on the top of each frame.

    Kt
    '''
    global tab_oscillographe_differences
    camera = function_initialize_camera(type_camera_number)
    cv.NamedWindow("WindowWebcam")
    width_frame  = int(640)
    height_frame = int(480)    
    
    # rectangle coordinates
    sub_rec1 = np.array([220,200,80,80])
    sub_rec2 = np.array([340,200,80,80])
    
    while True:
        frame = cv.QueryFrame(camera)
        # print frame
        # conversion to Lab
        image_filter = cv.CreateImage((width_frame, height_frame), cv.IPL_DEPTH_8U,3)
        cv.CvtColor(frame, image_filter,cv.CV_BGR2Lab)
        image_filter = frame

        frame, dL, dLab, LabT, LabB, dY, XYZT, XYZB, dRGB = function_display_live_information(frame, width_frame, height_frame, sub_rec1, sub_rec2)
        
        # Here we do something to display data difference as an osilloscope
        tab_oscillographe_differences[:,0:-1] = tab_oscillographe_differences[:,1:]
        tab_oscillographe_differences[:,-1] = [dL, dLab, dY, dRGB]

        frame = function_display_oscilloscope(frame, width_frame, height_frame, tab_oscillographe_differences)
        
        # --------------------------------------------
        cv.ShowImage("WindowWebcam",frame)
        k = cv.WaitKey(1)
        if k == ord('q'):
            print "Il faut sortir maintenant, faut pas rester ici."
            break

    return 1

def main():
    ''' The main function can gets as input parameter the camera number you want to 
    use.
    '''
    if len(sys.argv) == 1:
        print 'The camera number is 0.'
        function_show_webcam_stream()
    else:
        print 'The camera number is'+sys.argv[1]+'.'
        function_show_webcam_stream(int(sys.argv[1]))
    
      
# And now we start  
main()
print 'well done, you reach the biggest achievement of your day'
