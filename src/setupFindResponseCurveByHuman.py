'''
In that program we want to measure the response curve of a given display.

So we ask people with their tiny eyes to do it for us.

Jeremie Gerhardt - 24.01.14
'''

# Some modules to import
import cv2.cv as cv
import numpy as np
import scipy
from scipy import interpolate
from scipy.optimize import leastsq, curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import colorConverter
import Image
import time
from colorConversion import *
from colorDifferences import *
from webcamToos import *
import sys

# Some global variabl
mid_level = 128

# Some global variables to set for experiment 1 and 2
sizeTilePatchHT = 256  # parameter for the HT patches with halftoning by mask

vecLevel = np.round(np.hstack((np.arange(0, 255, 64), 255)))
stepVecSearch = 64
vecSearchLevel = np.round(np.hstack((np.arange(0,255, stepVecSearch), 255)))

prefixName = 'test35'
dirToSaveWebcamFrame = '/home/jeremie/Documents/workspace/devForWebcam/frameWebcam/'
dirToSaveResults = '/home/jeremie/Documents/workspace/devForWebcam/results/'


def function_get_response_curve_from_human(widthF,heightF):
    ''' The function does the same as funGetResponseCurve2 but instead of
    asking the camera we ask a human user to do the job

    Args:
        camera: an image but actually I don't need it here 
        levelContinuous: not needed
        levelHaltone: not needed
        widthF 
        heightF 
        sub_rec1: not needed
        sub_rec2: not needed

    Output:
        frame: an image  
        tabSelectedLevel: array (float [floats]) of size 1 x size(vecSearchLevel)

        tabDiffContinuousHalftoneL: array (float [floats]) of size 1 x size(vecSearchLevel)
        tabResultsContinousL      : array (float [floats]) of size 1 x size(vecSearchLevel)
        tabResultsHalftoneL       : array (float [floats]) of size 1 x size(vecSearchLevel)
        
    '''
    global vecLevel
    global vecSearchLevel
    global sizeTilePatchHT
    global tabOscillographeDifferences
    global max_number_frame_to_wait_between_measurement

    # imitialize some parameters
    tabSelectedLevel   = np.zeros((1,np.size(vecSearchLevel)))
    counterSearchLevel = 0

    # Some guidelines
    print 'q up to increase the level.'
    print 'a down to decrease the level.'
    print 'n to go to the next level.'

    for searchLevel in vecSearchLevel:
        # Here I create a testchart
        imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, 0, sizeTilePatchHT)
        imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
        imgT          = cv.fromarray(imgTestchart)
        cv.ShowImage("winTestChart",imgT)    

        print 'level haltoning is '+str(searchLevel)
        searchLevelC = 0

        while True:
            k_pressed = cv.WaitKey(5)
            # if keystroke pressed is 'q' then level up
            if k_pressed == ord('q'):
                searchLevelC = searchLevelC + 1
                if searchLevelC > 255:
                    searchLevelC = 255
                print 'level up '+str(searchLevelC)
                imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, searchLevelC, sizeTilePatchHT)
                imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
                imgT          = cv.fromarray(imgTestchart)
                cv.ShowImage("winTestChart",imgT)

            # if keystroke pressed is 'w' then level down
            elif k_pressed == ord('w'):
                print 'level down'
                searchLevelC = searchLevelC + 10
                if searchLevelC > 255:
                    searchLevelC = 255
                print 'level up '+str(searchLevelC)
                imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, searchLevelC, sizeTilePatchHT)
                imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
                imgT          = cv.fromarray(imgTestchart)
                cv.ShowImage("winTestChart",imgT)

            # if keystroke pressed is 'a' then level down
            elif k_pressed == ord('a'):
                print 'level down'
                searchLevelC = searchLevelC - 1
                if searchLevelC < 0:
                    searchLevelC = 0
                print 'level up '+str(searchLevelC)
                imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, searchLevelC, sizeTilePatchHT)
                imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
                imgT          = cv.fromarray(imgTestchart)
                cv.ShowImage("winTestChart",imgT)
            
            # if keystroke pressed is 's' then level down
            elif k_pressed == ord('s'):
                print 'level down'
                searchLevelC = searchLevelC - 10
                if searchLevelC < 0:
                    searchLevelC = 0
                print 'level up '+str(searchLevelC)
                imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, searchLevelC, sizeTilePatchHT)
                imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
                imgT          = cv.fromarray(imgTestchart)
                cv.ShowImage("winTestChart",imgT)

            # if keystroke pressed is 'e' the size tile change up
            elif k_pressed == ord('e'):
                print 'patch size tile down'
                sizeTilePatchHT = sizeTilePatchHT / 2
                if sizeTilePatchHT < 16:
                    sizeTilePatchHT = 16
                print 'level up '+str(sizeTilePatchHT)
                imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, searchLevelC, sizeTilePatchHT)
                imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
                imgT          = cv.fromarray(imgTestchart)
                cv.ShowImage("winTestChart",imgT)
            
            # if keystroke pressed is 'd' the size tile change down
            elif k_pressed == ord('d'):
                print 'patch size tile down'
                sizeTilePatchHT = sizeTilePatchHT * 2
                if sizeTilePatchHT > 512:
                    sizeTilePatchHT = 512   
                print 'level up '+str(sizeTilePatchHT)
                imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, searchLevelC, sizeTilePatchHT)
                imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
                imgT          = cv.fromarray(imgTestchart)
                cv.ShowImage("winTestChart",imgT)

            #if keystroke presses is n for "next" then next level
            elif k_pressed == ord('n'):
                tabSelectedLevel[0,counterSearchLevel] = searchLevelC # Here we save the Luminance of the left patch alone
                counterSearchLevel = counterSearchLevel + 1
                break

    print 'result by human observer:'
    print vecSearchLevel
    print tabSelectedLevel

    return tabSelectedLevel

def main():
    ''' Here the trouble start.
    '''

    global prefixName
    global configMeasurement
    global vecSearchLevel 
    global vecLevel
    global sizeTilePatchHT
    global stepVecSearch

    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)

    if sys.argv[1] == 'help':
        print ' '
        print 'The program functions as follows:'
        print 'python setupFindResponseCurveByHuman.py options'
        print 'where "options" are: name_test size_tile_patch_HT step_vector_search bool_experiment'
        print ' '
        print 'Some example to start an experiment:'
        print '     python setupFindResponseCurveAndRatio.py test33 256 64'
        print ' '
        
    else:
        prefixName = str(sys.argv[1])
        print prefixName

        sizeTilePatchHT = int(sys.argv[2])
        print sizeTilePatchHT

        stepVecSearch = int(sys.argv[3])
        print stepVecSearch
        vecSearchLevel = np.round(np.hstack((np.arange(0,255, stepVecSearch), 255)))
        vecSearchLevelExp3 = vecSearchLevel

        result_human_selection = function_get_response_curve_from_human(640 , 480)

        np.savetxt(dirToSaveResults+prefixName+'val_selected_by_user.txt', (result_human_selection),fmt='%03.2f')
        dataUser = np.loadtxt(dirToSaveResults+prefixName+'val_selected_by_user.txt')
        dataUser = np.vstack([vecSearchLevel, dataUser])
        function_display_RC_by_user(dataUser)
        plt.show()


main()  
print 'well done, you reach the biggest achievement of your day, or maybe the second, not bad.'
