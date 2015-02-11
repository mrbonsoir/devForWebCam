'''
This code is doing one thing: to measure the response curve of a display using your eyes or at least human eyes.
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import Image
import time
from colorConversion import *
from colorDifferences import *
from webcamTools import *
import sys, os, shutil

configTest = True
configMeasurement = False
number_camera = 0
max_number_frame_to_wait_between_measurement = 50
max_number_frame_to_keep  = 25
tabOscillographeDifferences = np.zeros((4,max_number_frame_to_keep))
max_number_point_to_show = 3
mid_level = 128
sizeTilePatchHT = 256  # parameter for the HT patches with halftoning by mask
dirToSaveResults = './'
workDir = os.getcwd()
selectExperiment_3 = False

vecLevelBasis = np.array([30,30, 30])       # Parameters for the ratio search
vecLevelTarget = 60                         #
vecLevelBasis2 = np.array([60, 60 , 60])    #
vecLevelTarget2 = 80                        #


widthFrame   = int(640)
heightFrame  = int(480)    

# config test
if configTest:
    prefixName         = 'humanbeta33'
    stepVecLevel       = 128
    vecLevel           = np.round(np.hstack((np.arange(0, 255, stepVecLevel), 255)))
    stepVecLevelSearch = 64
    vecSearchLevel     = np.round(np.hstack((np.arange(0,255, stepVecLevelSearch), 255)))

# config measurement
if configMeasurement:
    prefixName         = 'humantest33'
    stepVecLevel       = 64
    vecLevel           = np.round(np.hstack((np.arange(0, 255, stepVecLevel), 255)))
    stepVecLevelSearch = 32
    vecSearchLevel     = np.round(np.hstack((np.arange(0,255, stepVecLevelSearch), 255)))


def funDisplayWebcamAndTakePictures(number_camera): 
    '''
    The function is looping and waiting for you to choose which experiment to start.
    '''

    global tabOscillographeDifferences
    
    # call the webcam
    camera = function_initialize_camera(number_camera)
    
    # create the window to display the webcam stream and testchart
    cv.NamedWindow("Window")
    cv.NamedWindow("winTestChart")

    # get a first frame
    frame     = cv.QueryFrame(camera)
       
    # create Testchart
    widthFrame   = int(640)
    heightFrame  = int(480)    

    # rectangle coordinates
    sub_rec1 = np.array([220,200,80,80])
    sub_rec2 = np.array([340,200,80,80])
       
    # display one test-chart black/green to postion the window
    imgTestchart = imCreateTestchartContinuousAndHalftoned(mid_level,mid_level, sizeTilePatchHT)
    imgT = cv.CreateImage((np.shape(imgTestchart)[0],np.shape(imgTestchart)[0]), cv.IPL_DEPTH_8U,1)
    imgT = cv.fromarray(imgTestchart)
    cv.SaveImage('./firstTestChart.png', imgT)
    cv.ShowImage("winTestChart",imgT)

    ## create Testchart
    levelContinuous = mid_level
    levelHalftone   = mid_level

    # save the vecSearch and vecLevel for later use:
    np.savetxt('./'+prefixName+'_vecSearchLevel.txt', (vecSearchLevel),fmt='%03.2f') 
    np.savetxt('./'+prefixName+'_vecLevel.txt', (vecLevel),fmt='%03.2f')        
    
    # initialize table of data to show later the difference as oscilloscope
    print 's -> to start the measurement procedure.'
    print 'q -> to quit the measurement session.'

    # Start the loop
    while True:
        frame = cv.QueryFrame(camera)
        frame, dL, dLab, L1, L2, DY, XYZ1, XYZ2, dRGB = function_display_live_information(frame, widthFrame, heightFrame, sub_rec1, sub_rec2)
        cv.ShowImage("Window", frame)
        
        # Always capture the image displayed under the same name. Not very usefull so far.
        function_capture_image('imtestChartName_128_128.jpg',frame, './')
        # Here there is time to place the camera.

        # Here I start the loop for one channel as we did for CIC in 2012-------------------------------------
        k_pressed = cv.WaitKey(10)
        if k_pressed == ord("s"): # start the measurement if s is pressed
            val_selected_value_by_human1 = fun_Get_Ratio_By_Channel_By_Human(camera, widthFrame, heightFrame, sub_rec1, sub_rec2, vecLevelBasis, vecLevelTarget)
            
            np.savetxt(dirToSaveResults+prefixName+'_valSelected1.txt', (val_selected_value_by_human1),fmt='%03.2f')        

            # And we do it again
            print 'Same player with different input for basis and target'
            val_selected_value_by_human2 = fun_Get_Ratio_By_Channel_By_Human(camera, widthFrame, heightFrame, sub_rec1, sub_rec2, vecLevelBasis2, vecLevelTarget2)
            
            np.savetxt(dirToSaveResults+prefixName+'_valSelected2.txt', (val_selected_value_by_human2),fmt='%03.2f')        

        if k_pressed == ord('q'):
            print 'Il faut sortir maintenant, faut pas rester ici.'
            break

    # and the function does return nothing as every measurement are saved in a text file.        
    #return tabDiffL

def fun_Get_Ratio_By_Channel_By_Human(camera, 
                                      width_frame, height_frame, 
                                      sub_rec1, sub_rec2, 
                                      vecLevelBasis, vecLevelTarget):
    '''
    The function does like the function funGetRatioByChannel. Only difference is it asks
    a humna observer to decide if two levels are equivalent in intensity.

      Args:
        camera: an image but actually I don't need it here 
        levelContinuous: not needed
        levelHaltone: not needed
        widthF 
        heightF 
        sub_rec1: not needed
        sub_rec2: not needed

    Output:
        val_selected_value_by_human (array) : 1 x 3

        one value by color channel that tells where for which level there is equivalence
    '''
    global vecLevel
    global vecSearchLevel
    global sizeTilePatchHT
    global tabOscillographeDifferences
    global max_number_frame_to_wait_between_measurement

    # initialize some parameters
    tabSelectedLevel   = np.zeros((1,3))
        
    # create first testchart
    levelStep     = np.array([vecLevelBasis[0], vecLevelBasis[1], vecLevelBasis[2]])
    imgTestchart  = imCreateTestchartPatchBase(vecLevelTarget, levelStep)
    imgT          = cv.CreateImage((widthFrame,heightFrame), cv.IPL_DEPTH_8U,3)
    imgT          = cv.fromarray(imgTestchart)
    testChartName = 'imTestChartRatio_'+repr(vecLevelBasis[0])+'.jpg'
    cv.ShowImage("winTestChart",imgT)
    cv.SaveImage(dirToSaveResults+testChartName, imgT)
    cv.WaitKey(10)

    # Some guidelines
    print 'q/w  up to increase the level +1/+10.'
    print 'a/s  down to decrease the level -1/-10.'
    print 'n to go to the next channel.'

    searchLevelC = vecLevelBasis[0]
    # find the blue value    
    while True:
        k_pressed = cv.WaitKey(5)
        
        if k_pressed == ord('q'):
            searchLevelC = searchLevelC + 1
            if searchLevelC > 255:
                searchLevelC = 255
            print 'level up '+str(searchLevelC)

        if k_pressed == ord('w'):
            searchLevelC = searchLevelC + 10
            if searchLevelC > 255:
                searchLevelC = 255
            print 'level up '+str(searchLevelC)

        if k_pressed == ord('a'):
            searchLevelC = searchLevelC - 1
            if searchLevelC < 0:
                searchLevelC = 0
            print 'level down '+str(searchLevelC)

        if k_pressed == ord('s'):
            searchLevelC = searchLevelC - 10
            if searchLevelC < 0:
                searchLevelC = 0
            print 'level up '+str(searchLevelC)

        #if keystroke presses is n for "next" then next level
        elif k_pressed == ord('n'):
            tabSelectedLevel[0,0] = searchLevelC # Here we save the Luminance of the left patch alone

            print 'We are over with the blue channel.'
            imgTestchart  = imCreateTestchartPatchBase(vecLevelTarget, [vecLevelBasis[0], vecLevelBasis[1], vecLevelBasis[2]])
            imgT          = cv.CreateImage((width_frame, height_frame), cv.IPL_DEPTH_8U,3)
            imgT          = cv.fromarray(imgTestchart)
            cv.ShowImage("winTestChart",imgT)
            break

        imgTestchart  = imCreateTestchartPatchBase(vecLevelTarget, [searchLevelC, vecLevelBasis[1], vecLevelBasis[2]])
        imgT          = cv.CreateImage((width_frame, height_frame), cv.IPL_DEPTH_8U,3)
        imgT          = cv.fromarray(imgTestchart)
        cv.ShowImage("winTestChart",imgT)

    searchLevelC = vecLevelBasis[0]
    # find the green value    
    while True:
        k_pressed = cv.WaitKey(5)
        
        if k_pressed == ord('q'):
            searchLevelC = searchLevelC + 1
            if searchLevelC > 255:
                searchLevelC = 255
            print 'level up '+str(searchLevelC)

        if k_pressed == ord('w'):
            searchLevelC = searchLevelC + 10
            if searchLevelC > 255:
                searchLevelC = 255
            print 'level up '+str(searchLevelC)

        if k_pressed == ord('a'):
            searchLevelC = searchLevelC - 1
            if searchLevelC < 0:
                searchLevelC = 0
            print 'level down '+str(searchLevelC)

        if k_pressed == ord('s'):
            searchLevelC = searchLevelC - 10
            if searchLevelC < 0:
                searchLevelC = 0
            print 'level up '+str(searchLevelC)

        #if keystroke presses is n for "next" then next level
        elif k_pressed == ord('n'):
            tabSelectedLevel[0,1] = searchLevelC

            print 'We are over with the green channel.'
            imgTestchart  = imCreateTestchartPatchBase(vecLevelTarget, [vecLevelBasis[0], vecLevelBasis[1], vecLevelBasis[2]])
            imgT          = cv.CreateImage((width_frame, height_frame), cv.IPL_DEPTH_8U,3)
            imgT          = cv.fromarray(imgTestchart)
            cv.ShowImage("winTestChart",imgT)
            break

        imgTestchart  = imCreateTestchartPatchBase(vecLevelTarget, [vecLevelBasis[0], searchLevelC, vecLevelBasis[2]])
        imgT          = cv.CreateImage((width_frame, height_frame), cv.IPL_DEPTH_8U,3)
        imgT          = cv.fromarray(imgTestchart)
        cv.ShowImage("winTestChart",imgT)

    searchLevelC = vecLevelBasis[0]
    # find the red value    
    while True:
        k_pressed = cv.WaitKey(5)
        
        if k_pressed == ord('q'):
            searchLevelC = searchLevelC + 1
            if searchLevelC > 255:
                searchLevelC = 255
            print 'level up '+str(searchLevelC)

        if k_pressed == ord('w'):
            searchLevelC = searchLevelC + 10
            if searchLevelC > 255:
                searchLevelC = 255
            print 'level up '+str(searchLevelC)

        if k_pressed == ord('a'):
            searchLevelC = searchLevelC - 1
            if searchLevelC < 0:
                searchLevelC = 0
            print 'level down '+str(searchLevelC)

        if k_pressed == ord('s'):
            searchLevelC = searchLevelC - 10
            if searchLevelC < 0:
                searchLevelC = 0
            print 'level up '+str(searchLevelC)

        elif k_pressed == ord('n'):
            tabSelectedLevel[0,2] = searchLevelC

            print 'We are over with the red channel.'
            imgTestchart  = imCreateTestchartPatchBase(vecLevelTarget, [vecLevelBasis[0], vecLevelBasis[1], vecLevelBasis[2]])
            imgT          = cv.CreateImage((width_frame, height_frame), cv.IPL_DEPTH_8U,3)
            imgT          = cv.fromarray(imgTestchart)
            cv.ShowImage("winTestChart",imgT)
            break

        imgTestchart  = imCreateTestchartPatchBase(vecLevelTarget, [vecLevelBasis[0], vecLevelBasis[1], searchLevelC])
        imgT          = cv.CreateImage((width_frame, height_frame), cv.IPL_DEPTH_8U,3)
        imgT          = cv.fromarray(imgTestchart)
        cv.ShowImage("winTestChart",imgT)    


    print 'result by human observer: ', tabSelectedLevel

    return tabSelectedLevel

def funGetRatioByChannel(camera, widthF, heightF, sub_rec1, sub_rec2, valLevelBasis, valLevelTarget):
    '''
    Get the data to compute later the ratio.
    '''
    global tabOscillographeDifferences
       
    # intialiaze some stuffs
    tabResultDifferenceRGBL = np.zeros((3,np.size(vecSearchLevel)))
    tabResultDifferenceRGBY = np.zeros((3,np.size(vecSearchLevel)))
    tabResultBasisY  = np.zeros((3,np.size(vecSearchLevel)))
    tabResultTargetY = np.zeros((3,np.size(vecSearchLevel)))

    print vecSearchLevel
    for ii in np.arange(0,3):
        # start on channel
        counterSearchLevel = 0
        for level in vecSearchLevel:
            # Here I create a testchart for the blue channel
            if ii == 0:
                levelStep = np.array([valLevelBasis[0], valLevelBasis[1], level])
            elif ii == 1:
                levelStep = np.array([valLevelBasis[0], level, valLevelBasis[2]])
            elif ii == 2:
                levelStep = np.array([level, valLevelBasis[1], valLevelBasis[2]])
            else:
                print 'You are going too far.'

            imgTestchart  = imCreateTestchartPatchBase(valLevelTarget,levelStep)
            imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,3)
            imgT          = cv.fromarray(imgTestchart)
            testChartName = 'imTestChartRatioBlue_'+repr(level)+'.jpg'
            cv.ShowImage("winTestChart",imgT)
            cv.SaveImage(dirToSaveResults+testChartName, imgT)
            cv.WaitKey(10)

            # I display the stream again
            timer = 0
            while timer < max_number_frame_to_wait_between_measurement:
                frame = cv.QueryFrame(camera)
                frame, dL, dLab, LabT, LabB, dY, XYZT, XYZB, dRGB = function_display_live_information(frame, widthF, heightF, sub_rec1, sub_rec2)
                    
                # Here we do something to display data difference as an osilloscope
                tabOscillographeDifferences[:,0:-1] = tabOscillographeDifferences[:,1:]
                tabOscillographeDifferences[:,-1]   = [dL, dLab, dY, dRGB]
                frame = function_display_oscilloscope(frame, widthF, heightF, tabOscillographeDifferences)
                cv.ShowImage("Window", frame)
                timer = timer +1
                        
                cv.WaitKey(10)
                #if timer == max_number_frame_to_wait_between_measurement:
                    # Here I save the image taken by the webcam
                    #funCaptureImage('frameFor_'+testChartName,frame, dirToSaveWebcamFrame)

                # And here we save the differences. We will use them for computing the various ratio
                tabResultDifferenceRGBL[ii, counterSearchLevel]   = dL  # Here we save the difference
                tabResultDifferenceRGBY[ii, counterSearchLevel]   = dY  # Here we save the difference

                tabResultTargetY[ii, counterSearchLevel]   = LabT[0]      # Here we save the measured intensity
                tabResultBasisY[ii, counterSearchLevel]    = LabB[0]      # Here we save the measured intensity
            
            counterSearchLevel = counterSearchLevel + 1
                
    return frame, tabResultDifferenceRGBY, tabResultDifferenceRGBL, tabResultTargetY, tabResultBasisY

def funComputeRatioFromExperiment(dataRatioRGBY, dataRatioRGBL, valLevelBasis, valLevelTarget, dataTargetY, dataBasisY, titleText, figureName):
    '''
    The function first first display the results.
    Then it does some computations because computations are always good.
    '''
    # do some interpolation
    interpInput, interpY       =  funInterpolationRatioDifferenceCurves(vecSearchLevel, dataRatioRGBY)
    interpInput, interpTargetY =  funInterpolationRatioDifferenceCurves(vecSearchLevel, dataTargetY)
    interpInput, interpBasisY  =  funInterpolationRatioDifferenceCurves(vecSearchLevel, dataBasisY)
    
    # figure to show the search of equivalent intensity between the patches
    fig = plt.figure()
    yMin = 0
    yMax = 4.2*np.max(interpTargetY)
    #print yMax

    # plot the differences and minimum
    plt.plot(interpInput, interpY[0,:],'r-',label="Y difference Red ")
    plt.plot(interpInput, interpY[1,:],'g-',label="Y difference Green")
    plt.plot(interpInput, interpY[2,:],'b-',label="Y difference Blue")

    # plot the measured intensity
    plt.plot(interpInput, interpBasisY[0,:],'r--',label="Y Red + basis ")
    plt.plot(interpInput, interpBasisY[1,:],'g--',label="Y Green + basis")
    plt.plot(interpInput, interpBasisY[2,:],'b--',label="Y Blue + basis")

    # plot the target patch who should stay stable
    plt.plot(interpInput, interpTargetY[0,:],'k-',label="Y target for red ")
    plt.plot(interpInput, interpTargetY[1,:],'k--',label="Y target for green")
    plt.plot(interpInput, interpTargetY[2,:],'k-',label="Y target for blue")

    # plot the minimum
    minDiffInterpRGB, indRGB = funGetSomeMinimumSingleCurve(interpY)
    plt.plot(indRGB[0], minDiffInterpRGB[0],'r^')
    plt.plot(indRGB[1], minDiffInterpRGB[1],'g^')
    plt.plot(indRGB[2], minDiffInterpRGB[2],'b^')
    plt.vlines(indRGB[0],0,minDiffInterpRGB[0], colors='r',linestyles='--')
    plt.vlines(indRGB[1],0,minDiffInterpRGB[1], colors='g',linestyles='--')
    plt.vlines(indRGB[2],0,minDiffInterpRGB[2], colors='b',linestyles='--')

    # plot the experiment information
    plt.vlines(valLevelBasis[0], yMin, yMax, colors='k', linestyles='--', label='Basis')
    plt.vlines(valLevelTarget, yMin, yMax, colors='k', linestyles='--', label='Target')
    plt.text(valLevelBasis[0], yMax*0.9,'Basis = '+repr(valLevelBasis[0]), ha="center",bbox = dict(boxstyle='round', fc="w", ec="k"))
    plt.text(valLevelTarget, yMax*0.9,'Target = '+repr(valLevelTarget), ha="center",bbox = dict(boxstyle='round', fc="w", ec="k"))

    plt.ylabel('Difference in Y')
    plt.xlabel('Digital input')
    plt.xlim(0,255)
    plt.title('Difference Curve for Ratio')
    plt.ylim(yMin, yMax)
    plt.title(titleText)
    #plt.legend(loc=2)
    plt.draw()
    plt.savefig(figureName)

    print 'youou!!!'
    # I do nothing to start.
    ratioRGB = [indRGB[0], indRGB[1], indRGB[2]]
    return ratioRGB

def main():
    ''' Here the trouble start.
    '''
    global prefixName
    global configMeasurement
    global vecSearchLevel 
    global vecLevel
    global sizeTilePatchHT
    global stepVecSearch

    # do the measurement
    funDisplayWebcamAndTakePictures(number_camera)
    print 'The measurement sessions is closed.'
    print 'Now we process the data to get the ratios'

    # COMPUTE the response curve (RC) using Method 1:
    #ratioRGB1 = funComputeRatioFromExperiment(dataRatioRGB_Y, dataRatioRGB_L, vecLevelBasis, vecLevelTarget, dataResTarget_Y, dataResBasis_Y, 'config 1', dirToSaveResults+prefixName+'_ratio_Y1.png')
    #ratioRGB2 = funComputeRatioFromExperiment(dataRatioRGB_Y2, dataRatioRGB_L2, vecLevelBasis2, vecLevelTarget2, dataResTarget_Y2, dataResBasis_Y2, 'config 2', dirToSaveResults+prefixName+'_ratio_Y2.png')
    
    #plt.show()


main()  
print 'well done, you reach the biggest achievement of your day, or maybe the second, not bad.'
  