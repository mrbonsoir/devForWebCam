'''
In that program we want to measure the response curve of a given display.

To do so we compare halftoned and non-halftoned patches from which we know to input level.

Basically we do luminance matching and where it is becoming smart is we don't ask people to do it, we ask
the webcam gently attached to the Kompiouteur.

Jeremie Gerhardt - 26.06.13
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

# Some global variabl
number_camera = 1 # On my laptop, 0 is for the laptop cam, 1 is for the plugged cam
max_number_frame_to_wait_between_measurement = 50
max_number_frame_to_keep = 25
tabOscillographeDifferences = np.zeros((4,max_number_frame_to_keep))
max_number_point_to_show = 3
mid_level = 128

# Measurement or data analysis
configMeasurement = True

# Some global variables to set for experiment 1 and 2
selectExperiment_1 = False
selectExperiment_2 = False
selectExperiment_5 = False
sizeTilePatchHT = 256  # parameter for the HT patches with halftoning by mask

#vecLevel = np.round(np.linspace(0,255,8))       # To which step we want the measurement
#vecSearchLevel = np.round(np.linspace(0,255,12)) # To which step we want the point to get the difference curve

vecLevel = np.round(np.hstack((np.arange(0, 255, 64), 255)))
stepVecSearch = 64
vecSearchLevel = np.round(np.hstack((np.arange(0,255, stepVecSearch), 255)))
vecSearchLevelExp3 = vecSearchLevel#np.round(np.linspace(0,255,10))

'''
# Example for setting up the last two parameters:
# - I want to have measurement for a ramp space of 32 digital values so:
#>>> vecLevel = np.round(np.linspace(0,255,32))
# - and for each of this point I will take the measurement at space 16, so
#>>> vecSearchLevel = np.round(np.linspace(0,255,16))
# Ohter example, I want my RC made of 5 equally space point and measurement difference curve with twice the precision, so:
# vecLevel = np.round(np.linspace(0,255,5))       
# vecSearchLevel = np.round(np.linspace(0,255,10)) 
'''

# Some global variables to set for experiment 1 and 2
selectExperiment_3 = False
vecLevelBasis = np.array([30,30, 30])
vecLevelTarget = 60

vecLevelBasis2 = np.array([60, 60 , 60])
vecLevelTarget2 = 80

prefixName = str(sys.argv[1])
print 'Name of the experiment is '+prefixName
workDir = os.getcwd()
# create some directories to store the results
# test if the directory for storing the results already exist:

if len(sys.argv) > 2:
    if str(sys.argv[4]): # experiment have down already and we want just to recreate the figure
        if os.path.isdir(workDir+'/'+prefixName):
            print 'The folder '+workDir+' already exist.'
            print 'It''s going to be removed.'
         
            shutil.rmtree(workDir+'/'+prefixName)

            # For now it is a bit brutal because everything is removed
            os.mkdir(workDir+'/'+prefixName+'/')
            dirToSaveWebcamFrame = workDir+'/'+prefixName+'/frameWebcam/'
            os.mkdir(dirToSaveWebcamFrame)
            dirToSaveResults = workDir+'/'+prefixName+'/results/'
            os.mkdir(dirToSaveResults)

        else:
            os.mkdir(workDir+'/'+prefixName+'/')
            # I can create the folder and subfolder to store the results
            dirToSaveWebcamFrame = workDir+'/'+prefixName+'/frameWebcam/'
            os.mkdir(dirToSaveWebcamFrame)
            dirToSaveResults = workDir+'/'+prefixName+'/results/'
            os.mkdir(dirToSaveResults)

    else:
        print 'We look at the figures again without redoing the experimentself.'

def funDisplayWebcamAndTakePictures(number_camera): 
    '''
    The function is looping and waiting for you to choose which experiment to start.
    '''
    global selectExperiment_1
    global selectExperiment_2
    global selectExperiment_3
    global selectExperiment_5
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
    cv.SaveImage(dirToSaveWebcamFrame+'firstTestChart.png', imgT)
    cv.ShowImage("winTestChart",imgT)

    ## create Testchart
    levelContinuous = mid_level
    levelHalftone = mid_level

    # save the vecSearch and vecLevel for later use:
    np.savetxt('./'+prefixName+'_vecSearchLevel.txt', (vecSearchLevel),fmt='%03.2f') 
    np.savetxt('./'+prefixName+'_vecLevel.txt', (vecLevel),fmt='%03.2f')        
    np.savetxt('./'+prefixName+'_vecSearchLevelExp3.txt', (vecSearchLevelExp3),fmt='%03.2f')        


    # initialize table of data to show later the difference as oscilloscope
    # print the different options in the termninal
    print 'If you press theses keys 1, 2, 3, 4 or q then this will happend:'
    print '1 -> measurement starts according to your input parameter when calling the program.'
    print '2 -> same as 1 but faster where ramp of both continuous and haltoned patched are measured.'
    print '3 -> try to recover the ratio single channel vers grey.'
    print '4 -> here we suppose to get the maximum intensity per color channel.'
    print '5 -> here we do like in 1 or 2 but we ask a user to give his feedback.'
    print 'q -> is for Quit.'



    # Start the loop
    while True:
        frame = cv.QueryFrame(camera)
        frame, dL, dLab, L1, L2, DY, XYZ1, XYZ2, dRGB = function_display_live_information(frame, widthFrame, heightFrame, sub_rec1, sub_rec2)
        cv.ShowImage("Window", frame)
        
        # Always capture the image displayed under the same name. Not very usefull so far.
        function_capture_image('imtestChartName_128_128.jpg',frame, dirToSaveWebcamFrame)

        # Here I start the loop for one channel as we did for CIC in 2012-------------------------------------
        k_pressed = cv.WaitKey(10)
        if k_pressed == ord("1"): #cv.WaitKey(10) == 49: # i starts if "1" is pressed for measurement
            selectExperiment_1 = True
            frame, tabDiffL, tabDiffY = funGetResponseCurve(camera, widthFrame, heightFrame, sub_rec1, sub_rec2)
            print 'You can start over if you want.'

            # save all the measurement as text, this will use later to display and analayze the measurements
            np.savetxt(dirToSaveResults+prefixName+'_diffRC_L.txt', (tabDiffL),fmt='%03.2f')
            np.savetxt(dirToSaveResults+prefixName+'_diffRC_Y.txt', (tabDiffY),fmt='%03.2f')
            
        # Here I start a dumm approach where ramp in continuous tone and HT are diplayed and measured---------
        if k_pressed == ord("2"):#if cv.WaitKey(10) == 50: # for the letter "2"
            selectExperiment_2 = True
            frame, tabDiffL, tabContinuousL, tabHalftoneL = funGetResponseCurve2(camera, levelContinuous, levelHalftone, widthFrame, heightFrame, sub_rec1, sub_rec2)

            print 'You can start over if you want.'
            # save all the measurement as text
            np.savetxt(dirToSaveResults+prefixName+'_ContinuousRC_L.txt', (tabContinuousL),fmt='%03.2f')
            np.savetxt(dirToSaveResults+prefixName+'_HalftoneRC_L.txt', (tabHalftoneL),fmt='%03.2f')
            np.savetxt(dirToSaveResults+prefixName+'_diffContinuousHalftone_L.txt', (tabDiffL),fmt='%03.2f')

        # Here I start the method to recover the ratio color Vs grey-------------------------------------------
        if k_pressed == ord("3"):#if cv.WaitKey(10) == 51: # for the letter "3"
            selectExperiment_3 = True
            frame, tabDiffRGBY, tabDiffRGBL, tabResTargetY, tabResBasisY = funGetRatioByChannel(camera, widthFrame, heightFrame, sub_rec1, sub_rec2, vecLevelBasis, vecLevelTarget)
            
            # save all the measurement as text
            np.savetxt(dirToSaveResults+prefixName+'_diffRatioRGB_L.txt', (tabDiffRGBL),fmt='%03.2f')
            np.savetxt(dirToSaveResults+prefixName+'_diffRatioRGB_Y.txt', (tabDiffRGBY),fmt='%03.2f')        
            np.savetxt(dirToSaveResults+prefixName+'_diffResTarget_Y.txt', (tabResTargetY),fmt='%03.2f')        
            np.savetxt(dirToSaveResults+prefixName+'_diffResBasis_Y.txt', (tabResBasisY),fmt='%03.2f')        

            # And we do it again
            print 'Same player with different input for basis and target'
            frame, tabDiffRGBY, tabDiffRGBL, tabResTargetY, tabResBasisY = funGetRatioByChannel(camera, widthFrame, heightFrame, sub_rec1, sub_rec2, vecLevelBasis2, vecLevelTarget2)
            
            # save all the measurement as text
            np.savetxt(dirToSaveResults+prefixName+'_diffRatioRGB_L2.txt', (tabDiffRGBL),fmt='%03.2f')
            np.savetxt(dirToSaveResults+prefixName+'_diffRatioRGB_Y2.txt', (tabDiffRGBY),fmt='%03.2f')        
            np.savetxt(dirToSaveResults+prefixName+'_diffResTarget_Y2.txt', (tabResTargetY),fmt='%03.2f')        
            np.savetxt(dirToSaveResults+prefixName+'_diffResBasis_Y2.txt', (tabResBasisY),fmt='%03.2f')        

        # Here we start to get information about the maximum of each projector per channel
        if k_pressed == ord("4"):#if cv.WaitKey(10) == 52: # for the letter "4"
            frame, tabDiffRGBY, tabDiffRGBL, tabResTargetY, tabResBasisY = funGetRatioByChannel2(camera, widthFrame, heightFrame, sub_rec1, sub_rec2)

        # Here I start an approach where we ask user feedback instead of the camera
        if k_pressed == ord("5"):#if cv.WaitKey(10) == 53: # for the letter "5"
            selectExperiment_5 = True
            tabAnsweruser = function_get_response_curve_from_human(widthFrame, heightFrame)
            print tabAnsweruser
            print 'You can start over if you want. Just if you want.'
            # save all the measurement as text
            np.savetxt(dirToSaveResults+prefixName+'val_selected_by_user.txt', (tabAnsweruser),fmt='%03.2f')
            print selectExperiment_5
            
        frame, dL, dLab, LabT, LabB, dY, XYZT, XYZB, dRGB = function_display_live_information(frame, widthFrame, heightFrame, sub_rec1, sub_rec2)
        # Here we do something to display data difference as an osilloscope
        tabOscillographeDifferences[:,0:-1] = tabOscillographeDifferences[:,1:]
        tabOscillographeDifferences[:,-1] = [dL, dLab, dY, dRGB]
        frame = function_display_oscilloscope(frame, widthFrame, heightFrame, tabOscillographeDifferences)

        cv.ShowImage("Window", frame)        
        cv.ShowImage("winTestChart",imgT)
       
        '''
        If you press 'm' again it will restart the measurements and replace the existing figure and txt
        files from the previous session.
        '''

        if k_pressed == ord('q'):#cv.WaitKey(10) == 113: # if the letter 'q' is pressed then we quit.
            print 'Il faut sortir maintenant, faut pas rester ici.'
            break

    # and the function does return nothing as every measurement are saved in a text file.        
    #return tabDiffL

def funGetResponseCurve(camera, widthF, heightF, sub_rec1, sub_rec2):
    '''
    The function does things like loops, projection testchart and measurement.
    What it doesn't do is to provide the RC but only the data to compute it.
    '''
    global tabOscillographeDifferences

    # imitialize some parameters
    tabResultDifferenceL   = np.zeros((np.size(vecLevel),np.size(vecSearchLevel)))
    tabResultDifferenceY   = np.zeros((np.size(vecLevel),np.size(vecSearchLevel)))
    print 'data to be saved of size ',np.shape(tabResultDifferenceL)
    print vecLevel
    counterLevel, counterSearchLevel = 0, 0
    for level in vecLevel:
        for searchLevel in vecSearchLevel:
            #print repr(level)+'/'+repr(searchLevel)
            # Here I create a testchart
            imgTestchart  = imCreateTestchartContinuousAndHalftoned(level, searchLevel, sizeTilePatchHT)
            imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
            imgT          = cv.fromarray(imgTestchart)
            cv.ShowImage("winTestChart",imgT)    
            #testChartName = 'imTestChart_'+repr(level)+'_'+repr(searchLevel)+'.png'
            testChartName = 'imTestChart_'+repr(level)+'_'+repr(searchLevel)+'.jpg'
            cv.SaveImage(dirToSaveWebcamFrame+testChartName, imgT)
            cv.ShowImage("winTestChart",imgT)
            cv.WaitKey(10)

            # I display the stream again
            timer = 0
            while timer < max_number_frame_to_wait_between_measurement:
                frame = cv.QueryFrame(camera)
                frame, dL, dLab, LabT, LabB, dY, XYZT, XYZB, dRGB = function_display_live_information(frame, widthF, heightF, sub_rec1, sub_rec2)
                
                # Here we do something to display data difference as an osilloscope
                tabOscillographeDifferences[:,0:-1] = tabOscillographeDifferences[:,1:]
                tabOscillographeDifferences[:,-1] = [dL, dLab, dY, dRGB]
                frame = function_display_oscilloscope(frame, widthF, heightF, tabOscillographeDifferences)
                cv.ShowImage("Window", frame)
                timer = timer +1
                    
                cv.WaitKey(5)
                
                #if timer == max_number_frame_to_wait_between_measurement:
                    # Here I save the image taken by the webcam
                    #funCaptureImage('frameFor_'+testChartName,frame, dirToSaveWebcamFrame)

            # And here we save the differences. We will use them for computing the various ratio
            tabResultDifferenceL[counterLevel, counterSearchLevel]   = dL      # Here we save the difference
            tabResultDifferenceY[counterLevel, counterSearchLevel]   = dY      # Here we save the difference
            counterSearchLevel = counterSearchLevel + 1
            print repr(counterLevel)+'/'+repr(counterSearchLevel)

        counterLevel = counterLevel + 1
        counterSearchLevel = 0

        # remove the offset


    return frame, tabResultDifferenceL, tabResultDifferenceY

def funGetResponseCurve2(camera, levelContinuous, levelHaltone, widthF, heightF, sub_rec1, sub_rec2):
    '''
    The function does things like measuring response curve of continuous tone and halftone ramp as the
    webcam was a proper measuring device, yes you read it right MF.
    '''
    global vecLevel
    global vecSearchLevel
    global sizeTilePatchHT
    global tabOscillographeDifferences
    global max_number_frame_to_wait_between_measurement

    # imitialize some parameters
    tabResultsContinousL       = np.zeros((1,np.size(vecSearchLevel)))
    tabResultsHalftoneL        = np.zeros((1,np.size(vecSearchLevel)))
    tabDiffContinuousHalftoneL = np.zeros((1,np.size(vecSearchLevel)))
    print 'data to be saved of size ',np.shape(tabResultsContinousL)
    
    counterSearchLevel = 0
    for searchLevel in vecSearchLevel:
        # Here I create a testchart
        imgTestchart  = imCreateTestchartContinuousAndHalftoned(searchLevel, searchLevel, sizeTilePatchHT)
        imgT          = cv.CreateImage((widthF,heightF), cv.IPL_DEPTH_8U,1)
        imgT          = cv.fromarray(imgTestchart)
        cv.ShowImage("winTestChart",imgT)    
        #testChartName = 'imTestChart_'+repr(searchLevel)+'_'+repr(searchLevel)+'.png'
        testChartName = 'imTestChart_'+repr(searchLevel)+'_'+repr(searchLevel)+'.jpg'
        cv.SaveImage(dirToSaveWebcamFrame+testChartName, imgT)
        cv.ShowImage("winTestChart",imgT)
        cv.WaitKey(10)

        # I display the stream again
        timer = 0
        while timer < max_number_frame_to_wait_between_measurement:
        #for i in np.arange(0,50):
            #print '\ntemp frame ' + repr(timer) + '/24.'
            frame = cv.QueryFrame(camera)
            frame, dL, dLab, LabT, LabB, dY, XYZT, XYZB, dRGB = function_display_live_information(frame, widthF, heightF, sub_rec1, sub_rec2)
                                
            # Here we do something to display data difference as an osilloscope
            tabOscillographeDifferences[:,0:-1] = tabOscillographeDifferences[:,1:]
            tabOscillographeDifferences[:,-1] = [dL, dLab, dY, dRGB]

            frame = function_display_oscilloscope(frame, widthF, heightF, tabOscillographeDifferences)
            cv.ShowImage("Window", frame)
            timer = timer +1

            cv.WaitKey(5)
            #if timer == max_number_frame_to_wait_between_measurement:
                # Here I save the image taken by the webcam
                #funCaptureImage('frameFor_'+testChartName,frame, dirToSaveWebcamFrame)

        # And here we save the differences. We will use them for computing the various ratio
        tabResultsContinousL[0,counterSearchLevel]       = LabT[0] # Here we save the Luminance of the left patch alone
        tabResultsHalftoneL[0,counterSearchLevel]        = LabB[0] # Here we save the Luminace of the right patch alone
        tabDiffContinuousHalftoneL[0,counterSearchLevel] = dL      # Here we save the difference
        counterSearchLevel = counterSearchLevel + 1
        #print counterSearchLevel

    return frame, tabDiffContinuousHalftoneL, tabResultsContinousL, tabResultsHalftoneL

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
        

    pseudo code:
    display testChart 
        increase level in blue channel until satisfying
            go to next channel
        increase level in red channel until satisfying
            go to next channel 
        increase level in green channel until satisfying
            go to next round and restart with new ref and basis value


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

def funGetRatioByChannel(camera, widthF, heightF, sub_rec1, sub_rec2, valLevelBasis, valLevelTarget):
    '''
    Get the data to compute later the ratio.
    '''
    global tabOscillographeDifferences
       
    # intialiaze some stuffs
    tabResultDifferenceRGBL = np.zeros((3,np.size(vecSearchLevelExp3)))
    tabResultDifferenceRGBY = np.zeros((3,np.size(vecSearchLevelExp3)))
    tabResultBasisY  = np.zeros((3,np.size(vecSearchLevelExp3)))
    tabResultTargetY = np.zeros((3,np.size(vecSearchLevelExp3)))

    print vecSearchLevelExp3
    for ii in np.arange(0,3):
        # start on channel
        counterSearchLevel = 0
        for level in vecSearchLevelExp3:
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
            cv.SaveImage(dirToSaveWebcamFrame+testChartName, imgT)
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

def funGetRatioByChannel2(camera, widthF, heightF, sub_rec1, sub_rec2):
    '''
    Get the data to compute later the ratio. But this time we do it smarter. We first
    display some maximum Red, Blue and Green and we get an idea of the maximum of basis 
    value we can use for the search of Target = Basis.
    '''
    global tabOscillographeDifferences
    valLevelBasis = 0
    valLevelTarget = 0
       
    # intialiaze some stuffs
    tabConfig = np.array([[0,0,255],[0,255,0],[255,0,0],[255,255,255]])
    tabMaxIntensityMaxRGBY = np.zeros((1,4))
    tabResultDifferenceRGBL = np.zeros((1,4))#np.zeros((3,np.size(vecSearchLevelExp3)))
    tabResultDifferenceRGBY = np.zeros((1,4))#np.zeros((3,np.size(vecSearchLevelExp3)))
    tabResultBasisY  = np.zeros((3,np.size(vecSearchLevelExp3)))
    tabResultTargetY = np.zeros((3,np.size(vecSearchLevelExp3)))
    print tabResultDifferenceRGBL, np.shape(tabResultDifferenceRGBL)

    # first we display only the primaries, each one alone and we deduce the maximum
    for ii in np.arange(0,4):
        print tabConfig[ii,:], ii
        imgT = cv.fromarray(imCreateTestChartSingleColor(tabConfig[ii,:]))
        cv.ShowImage("winTestChart",imgT)
        cv.WaitKey(20)

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
            print timer
                            
        cv.WaitKey(10)
        tabResultDifferenceRGBL[0,ii]  = dL      # Here we save the difference
        tabResultDifferenceRGBY[0,ii]  = dY      # Here we save the difference
        tabMaxIntensityMaxRGBY[0,ii]   = (LabT[0] + LabB[0]) / 2.     # Here we save the measured intensity


    print 'L difference: ',tabResultDifferenceRGBL
    print 'Y difference: ',tabResultDifferenceRGBY
    print 'max L RGB: ', tabMaxIntensityMaxRGBY
                
    return frame, tabResultDifferenceRGBY, tabResultDifferenceRGBL, tabResultTargetY, tabResultBasisY

def funComputeResponseCurveFromMeasurement(tabMeasurement, vectorLevel, vectorSearchLevel):
    '''
    The function does some processing like interpolation on the obtained curves, this to
    find the local minima corresponding to the search value for a given digital input.
    '''
    '''
    # remove the offset:
    print np.shape(vecSearchLevel)
    print np.shape(tabMeasurement)
    for ii in np.arange(0,np.size(vecLevel)):
        offset = np.min(tabMeasurementContinuous[ii,:])
        tabMeasurementContinuous[ii,:]=tabMeasurementContinuous[ii,:]-offset
    '''    
    #print 'measurement shape :',np.shape(tabMeasurement)
    #print 'measurement  size :',np.shape(vecSearchLevel)
    #print vecSearchLevel
    #print tabMeasurement
    #print 'vecLevel shape    :',np.shape(vecSearchLevel)
    
    # Some interpolation for the Y measurements of target and basis
    interpInput, interpM = funInterpolationSingleCurve(np.vstack([vectorSearchLevel,tabMeasurement]))
    # and we normaliez the bazard
    interpM  = interpM  / np.max(interpM)

    # figure to show the search of equivalent intensity between the patches
    responseCurve = funDisplayDifferenceCurveAndMinimun(vectorLevel, interpInput, interpM,'Digital input','L Difference','Various difference curves method 1',dirToSaveResults+prefixName+'_c1_1.png')
    responseCurve[0]  = 0   # to set 0 to 0 
    responseCurve[-1] = 255 # to set last value to the maximum

    # the same as above but in "3D"
    funDisplayDifferenceCurveIn3D(vectorLevel, interpInput, interpM,'Digital input','Level tested','L Difference','Various difference curves method 1',dirToSaveResults+prefixName+'_c1_2.png')
    # plot response curve
    #funPlotOneResponseCurves(vecLevel, responseCurve, 'mehtod 1', dirToSaveResults+prefixName+'_c1_3.png')

    return responseCurve

def funComputeResponseCurveFromMeasurementExp2(tabMeasurementContinuous, tabMeasurementHalftone, vectorSearchLevel):
    '''
    The function does some processing like interpolation on the obtained curves, this to
    find the local minima corresponding to the search value for a given digital input.
    '''
    # do something for the offset:
    tabMeasurementHalftone = tabMeasurementHalftone - tabMeasurementHalftone[0]
    tabMeasurementContinuous = tabMeasurementContinuous - tabMeasurementContinuous[0] 

    # Some interpolation for the Y measurements of target and basis
    interpInput, interpMC, interpMHT =  funInterpolationContinuousHalftonecurve(vectorSearchLevel, tabMeasurementContinuous, tabMeasurementHalftone)
    interpMC  = interpMC  / np.max(interpMC)
    interpMHT = interpMHT / np.max(interpMHT) # and we normalize the bazard
    
    # and we compute the difference
    interpDataDiff = np.zeros((np.size(vectorSearchLevel),np.size(interpInput)))
    ii = 0
    for level in vectorSearchLevel:
        Y_fc = interpMC[level]
        interpDataDiff[ii,:] = np.sqrt((interpMHT - Y_fc)*(interpMHT - Y_fc))
        ii = ii + 1
    
    # now we build the curve based on these two curves:
    dataResponseCurve = np.zeros(np.size(vectorSearchLevel))
    dataResponseCurve = funDisplayDifferenceCurveAndMinimun(vectorSearchLevel, interpInput, interpDataDiff, 'Digital input','L Difference','Various difference curves method 2', dirToSaveResults+prefixName+'_c2_1.png')
    dataResponseCurve[0]  = 0   # to set 0 to 0 
    dataResponseCurve[-1] = 255 # to set last value to the maximum
    
    # the same in 3D
    #funDisplayDifferenceCurveIn3D(vectorSearchLevel, interpInput, interpDataDiff,'Digital input','Level tested','L Difference','Various difference curves method 2',dirToSaveResults+prefixName+'_c2_2.png')

    # plot response curve and measurement
    funPlotMeasurementCurves(interpInput, interpMC, interpMHT, vectorSearchLevel, dataResponseCurve/np.max(dataResponseCurve), dirToSaveResults+prefixName+'_c2_4.png')
   
    return interpMC , interpMHT, dataResponseCurve

def funPlotMeasurementCurves(interpInput, interpMC, interpMHT, vecInputRC, dataRC, figureName):
    '''
    Function to show the response curves measured from HT and continuous patches and the computed
    response curve we obtain.
    '''
    fig = plt.figure()
    plt.plot(interpInput, interpInput / 255.,'k-', label='linear')
    plt.plot(np.power(interpInput / 255.,2.2),'k-', label='gamma 2.2')
    plt.plot(interpInput, interpMC,'r',label='continuous Y')
    plt.plot(interpInput, interpMHT,'b',label='haltoned Y')
    plt.plot(vecInputRC, dataRC,'g',label='response curve')
    plt.xlabel('Digital input')
    plt.xlim(0,255)
    plt.ylim(0,1)
    plt.legend(loc=2)
    plt.draw()
    plt.savefig(figureName)

def funComputeRatioFromExperiment(dataRatioRGBY, dataRatioRGBL, valLevelBasis, valLevelTarget, dataTargetY, dataBasisY, titleText, figureName):
    '''
    The function first first display the results.
    Then it does some computations because computations are always good.
    '''
    # do some interpolation
    interpInput, interpY       =  funInterpolationRatioDifferenceCurves(vecSearchLevelExp3, dataRatioRGBY)
    interpInput, interpTargetY =  funInterpolationRatioDifferenceCurves(vecSearchLevelExp3, dataTargetY)
    interpInput, interpBasisY  =  funInterpolationRatioDifferenceCurves(vecSearchLevelExp3, dataBasisY)
    
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

def funPlotOneResponseCurves(vecInput, responseCurve1, methodName, figureName):
    '''
    Here we plot the response curve obtained by one method only.
    '''
    interpInput = np.arange(0,256)
    fig = plt.figure()
    plt.plot(interpInput, interpInput / 255.,'k-')
    plt.plot(np.power(interpInput / 255.,2.2),'k-')
    plt.text(255 / 2,                  0.5, 'linear', horizontalalignment='center', verticalalignment='center', rotation=38,bbox = dict(boxstyle='round', fc="w", ec="k"))
    plt.text(180, np.power(170 / 255.,2.2), 'gamma 2.2', horizontalalignment='center', verticalalignment='center', rotation=48.,bbox = dict(boxstyle='round', fc="w", ec="k"))
    
    plt.plot(vecInput,  responseCurve1 / 255.,'b-', label=methodName)
    plt.plot(vecInput,  responseCurve1 / 255.,'.b')
    
    plt.xlabel('Digital input')
    plt.title('Response curve by '+methodName)
    plt.xlim(0,255)
    plt.ylim(0,1)
    plt.legend(loc=2)
    plt.draw()
    plt.savefig(figureName)

def funPlotTwoResponseCurves(vecInput1, responseCurve1, vecInput2, responseCurve2, figureName):
    '''
    Here we plot the response curve obtained by the two first methods
    '''
    interpInput = np.arange(0,256)
    fig = plt.figure()
    plt.plot(interpInput, interpInput / 255.,'k-')
    plt.plot(np.power(interpInput / 255.,2.2),'k-')
    plt.text(255 / 2,                  0.5, 'linear', horizontalalignment='center', verticalalignment='center', rotation=38,bbox = dict(boxstyle='round', fc="w", ec="k"))
    plt.text(180, np.power(170 / 255.,2.2), 'gamma 2.2', horizontalalignment='center', verticalalignment='center', rotation=48.,bbox = dict(boxstyle='round', fc="w", ec="k"))
    
    plt.plot(vecInput1, responseCurve1 / 255.,'b-', label='method 1')
    plt.plot(vecInput2, responseCurve2 / 255.,'r-',label='method 2')
    plt.plot(vecInput1, responseCurve1 / 255.,'.b')
    plt.plot(vecInput2, responseCurve2 / 255.,'r.')
    
    plt.xlabel('Digital input')
    plt.title('Response Curve comparison')
    plt.xlim(0,255)
    plt.ylim(0,1)
    plt.legend(loc=2)
    plt.draw()
    plt.savefig(figureName)   

def funDisplayDifferenceCurveAndMinimun(vecDigitLevel, inputData_x, dataToDisplay_y, xLabelText, yLabelText, titleText, figureName):
    '''
    The function displays the dataToDisplay_y and put a red point on the minimum
    of each curve.
    Eventually it returns the computed response curve.
    '''
    #print ' hello you', np.shape(vecDigitLevel)
    minDataToDisplay, indMinimum = funGetSomeMinimumSingleCurve(dataToDisplay_y)
    fig = plt.figure()
    for ii in np.arange(0,np.size(vecDigitLevel)):
        plt.plot(inputData_x, dataToDisplay_y[ii,:],'b-')
    plt.plot(indMinimum, minDataToDisplay,'ro')
    plt.xlabel(xLabelText)
    plt.ylabel(yLabelText)
    plt.xlim(0,255)
    plt.ylim(0,1)
    plt.title(titleText)
    plt.draw()
    plt.savefig(figureName)

    return indMinimum

def funDisplayDifferenceCurveIn3D(vecDigitLevel, inputData_x, dataToDisplay_y, xLabelText, yLabelText, zLabelText, titleText, figureName):
    '''
    Exactly the same as the function above, but in 3D, yes in 3D, it is the future here.
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.2)

    xs = inputData_x
    verts = []
    tabColor = []
    zs = vecDigitLevel
    for ii in np.arange(0,np.size(vecDigitLevel)):
        ys = dataToDisplay_y[ii,:]    
        ys[0], ys[-1] = 0, 0  
        verts.append(list(zip(xs, ys)))
        tabColor.append(list(cc(repr(vecDigitLevel[ii]/255.))))

    poly = PolyCollection(verts, facecolors = tabColor)
    poly.set_alpha(0.7)

    ax.add_collection3d(poly, zs=zs, zdir='y')
    ax.set_xlabel(xLabelText)#'level search')
    ax.set_xlim3d(0, 255)
    ax.set_ylabel(yLabelText)#'level tested')
    ax.set_ylim3d(-1, 256)
    ax.set_zlabel(zLabelText)#L difference')
    ax.set_zlim3d(0, 1)
    plt.title(titleText)#'Various difference curves in 3D')
    plt.draw()
    plt.savefig(figureName)# dirToSaveResults+prefixName+'_c1_2.png')

def funComputeFittedResponseCurve(responseCurve, vec_meas_curve, data_stellar):
    '''
    The function takes the points from the computed response curve and fit a curve to get something who look like
    a gamma kind of shape
    '''
    #def fitFunc(t, A, B, alpha, beta):
    #    nume = A*np.power(t/255., alpha) 
    #    deno = np.power(t/255., beta) + B
    #    deno = deno + 1e-6 # ugly trick
    #    val = nume / deno  
    #    # and normalization des familles
    #    val = val / ( (A*np.power(1, alpha)) / (np.power(1, beta) + B))
    #    return val

    x = vec_meas_curve
    gamma22 = funSuperFittFunction(x, 1, 0, 2.2,0)
    y_meas = responseCurve
    y_meas = y_meas / np.max(y_meas)
    #print y_meas

    def residuals(p, y, x):
        A, B, alpha, beta = p
        err = y - funSuperFittFunction(x, A, B, alpha, beta)
        return err

    def peval(x, p):
        A, B, alpha, beta = p
        res = funSuperFittFunction(x, A, B, alpha, beta)
        return res

    p0 = [1, 0, 2, 0]

    # do the optimization biatch
    plsq = leastsq(residuals, p0, args=(y_meas, x))
    fitted_Response_Curve = peval(x,plsq[0])
    #print np.shape(plsq), plsq[0]

    # OR WE DO IT LIKE THIS:
    estimated_param, err_estim = curve_fit(funSuperFittFunction, x, y_meas, p0)
    fitted_Response_Curve2 = peval(x,estimated_param)
    print 'Hammer time----------------------------------------'
    print 'ICI REGARDE'
    print 'curve fit',estimated_param, np.shape(estimated_param)
    print 'lseastsq', plsq[0], np.shape(plsq[0])
    print 'Hammer time----------------------------------------'
    #fitted_Response_Curve = funSuperFittFunction(x, estimated_param)

    guess_start_curve = peval(x,p0)
    # display the results
    fig = plt.figure()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #plt.xlabel(r'\textbf{time} (s)')
    #plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
    #plt.title(r"\TeX\ is Number "r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",fontsize=16, color='gray')
    print guess_start_curve
    plt.plot(x,guess_start_curve,'k',label='guess start')
    plt.plot(x,gamma22,'r',label='gamma 2.2')
    plt.plot(x,y_meas,'g.-', label ='measurement')
    print np.shape(data_stellar)
    plt.plot(data_stellar[:,0], data_stellar[:,1],'.-k',label='Spectro')

    A_par, B_par, alpha_par, beta_par = plsq[0]
    print A_par, B_par, alpha_par, beta_par

    labelTex = r"Mac Fitt "r"$\displaystyle A\frac{x^\alpha}{x^\beta + C}$"
    plt.plot(x,fitted_Response_Curve,'b' ,label= labelTex)
    plt.title('Least-squares fit to noisy data')
    plt.legend(loc='upper left')
    plt.xlim(0,255)
    plt.ylim(0,1)
    plt.draw()
    figureName = dirToSaveResults+prefixName+'_c2_5.png'
    plt.savefig(figureName)
    #plt.show()

    return fitted_Response_Curve, plsq[0]

def funGiveMeTheRatio(ratio_RGB, basisRGB, parameters_gamma_curve):
    '''
    This function finally does things.
    We d0 -> Y_color_Channel_relative - Y_color_basis
    '''
    # get effective values from the RC curve function
    A, B, alpha, beta = parameters_gamma_curve
    Y_R_relatif = funSuperFittFunction(ratio_RGB[0], A, B, alpha, beta)
    Y_B_relatif = funSuperFittFunction(ratio_RGB[1], A, B, alpha, beta)
    Y_G_relatif = funSuperFittFunction(ratio_RGB[2], A, B, alpha, beta)
    Y_Basis_relatif = funSuperFittFunction(basisRGB, A, B, alpha, beta)
    ratio_GR =  (Y_G_relatif - Y_Basis_relatif) / (Y_R_relatif - Y_Basis_relatif)
    ratio_GB = (Y_G_relatif - Y_Basis_relatif) / (Y_B_relatif - Y_Basis_relatif)

    print 'Some results----------------------------- below by our method'
    print 'pure R '+str(ratio_RGB[0])+' give Y R relative of '+str(Y_R_relatif)
    print 'pure G '+str(ratio_RGB[1])+' give Y R relative of '+str(Y_G_relatif)
    print 'pure B '+str(ratio_RGB[2])+' give Y R relative of '+str(Y_B_relatif)
    print 'pure Basis '+str(basisRGB)+' give Y R relative of '+str(Y_Basis_relatif)
    print 'ratio GR: ', ratio_GR
    print 'ratio GB: ', ratio_GB    
    return Y_R_relatif, Y_G_relatif, Y_B_relatif, ratio_GR, ratio_GB

def funGiveMeTheRatioBySpectro(ratio_RGB, basisRGB, data_response_curve_stellar):
    '''
    This function finally does things.
    We d0 -> Y_color_Channel_relative - Y_color_basis
    '''
    # get effective values from the RC curve function by linear interpolation
    Y_R_relatif = np.interp(ratio_RGB[0], data_response_curve_stellar[:,0], data_response_curve_stellar[:,1])
    Y_B_relatif = np.interp(ratio_RGB[1], data_response_curve_stellar[:,0], data_response_curve_stellar[:,1])
    Y_G_relatif = np.interp(ratio_RGB[2], data_response_curve_stellar[:,0], data_response_curve_stellar[:,1])
    Y_Basis_relatif = np.interp(basisRGB, data_response_curve_stellar[:,0], data_response_curve_stellar[:,1])
    ratio_GR =  (Y_G_relatif - Y_Basis_relatif) / (Y_R_relatif - Y_Basis_relatif)
    ratio_GB = (Y_G_relatif - Y_Basis_relatif) / (Y_B_relatif - Y_Basis_relatif)

    # get effective values from the RC curve by Stellare device

    print 'Some results----------------------------- below by Stellar'
    print 'pure R '+str(ratio_RGB[0])+' give Y R relative of '+str(Y_R_relatif)
    print 'pure G '+str(ratio_RGB[1])+' give Y R relative of '+str(Y_G_relatif)
    print 'pure B '+str(ratio_RGB[2])+' give Y R relative of '+str(Y_B_relatif)
    print 'pure Basis '+str(basisRGB)+' give Y R relative of '+str(Y_Basis_relatif)
    print 'ratio GR: ', ratio_GR
    print 'ratio GB: ', ratio_GB    
    return Y_R_relatif, Y_G_relatif, Y_B_relatif, ratio_GR, ratio_GB

def main():
    ''' Here the trouble start.
    '''

    global selectExperiment_1
    global selectExperiment_2
    global selectExperiment_3
    global selectExperiment_5
    global prefixName
    global configMeasurement
    global vecSearchLevel 
    global vecLevel
    global sizeTilePatchHT
    global stepVecSearch
    global vecSearchLevelExp3
    #global dirToSaveWebcamFrame

    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)

    if sys.argv[1] == 'help':
        print ' '
        print 'The program functions as follows:'
        print 'python setupFindResponseCurveAndRatio.py options'
        print 'where "options" are: [name_test] [size_tile_patch_HT] [step_vector_search bool_experiment] [camera_number]'
        print ' '
        print 'Some example to start an experiment:'
        print '     python setupFindResponseCurveAndRatio.py test33 256 64 True 1'
        print 'and this if you just want to re-generate curve and graphic from the measurement:'
        print '     python setupFindResponseCurveAndRatio.py test33 256 64 False'
        print 'Time to try again...'
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

        argvConfigMeasurement = sys.argv[4]
        if argvConfigMeasurement == 'False':
            configMeasurement = False

            print "we are here NOT for measurement"
            selectExperiment_1 = False
            selectExperiment_2 = True
            selectExperiment_3 = True
            selectExperiment_5 = True

        if argvConfigMeasurement == 'True':
            configMeasurement = True 

            if len(sys.argv) == 6:
                camera_number = sys.argv[5]

            # start the measurements
            if configMeasurement:
                funDisplayWebcamAndTakePictures(int(sys.argv[5]))
                print "we are here for measurement"     

        vecLevel = np.loadtxt(dirToSaveResults+prefixName+'_vecLevel.txt')
        vecSearchLevel = np.loadtxt(dirToSaveResults+prefixName+'_vecSearchLevel.txt')

    # load results from experiment "1"
    if selectExperiment_1:
        # The data have been saved when the experiment 1 has been selected.
        dataDiff  = np.loadtxt(dirToSaveResults+prefixName+'_diffRC_L.txt')   

        responseCurve = funComputeResponseCurveFromMeasurement(dataDiff, vecLevel, vecSearchLevel)

    # load results from experiment "2"
    if selectExperiment_2:
        # load data obtained with the Stellar device:
        print 'data ramp', vecSearchLevel
        dataRCbyStellar = np.loadtxt(dirToSaveResults+'dataResponseCurveOptomaStellar.txt')
        print np.shape(dataRCbyStellar)

        # The data have been saved when the experiment 2 has been selected.
        dataContinuous = np.loadtxt(dirToSaveResults+prefixName+'_ContinuousRC_L.txt')
        dataHalftone   = np.loadtxt(dirToSaveResults+prefixName+'_HalftoneRC_L.txt')
        dataDiff       = np.loadtxt(dirToSaveResults+prefixName+'_diffContinuousHalftone_L.txt')
        responseCurveContinuous, responseCurveHaltone, responseCurve2 = funComputeResponseCurveFromMeasurementExp2(dataContinuous, dataHalftone, vecSearchLevel)
                
        # save response curve maybe?
        np.savetxt(dirToSaveResults+prefixName+'_RC_continuous.txt', (responseCurveContinuous),fmt='%03.2f')
        np.savetxt(dirToSaveResults+prefixName+'_RC_halftone.txt', (responseCurveHaltone),fmt='%03.2f')        
        np.savetxt(dirToSaveResults+prefixName+'_RC_final.txt', (responseCurve2),fmt='%03.2f')  

        # and now do some processing to fit a curve on the obtained RC
        dataResponseCurve = np.loadtxt(dirToSaveResults+prefixName+'_RC_final.txt')  
        fittedResponseCurve, parameters_gamma = funComputeFittedResponseCurve(dataResponseCurve, vecSearchLevel, dataRCbyStellar)
        np.savetxt(dirToSaveResults+prefixName+'_RC_final_fitted.txt', (responseCurve2),fmt='%03.2f') 
        np.savetxt(dirToSaveResults+prefixName+'_param_gamma_curve.txt', (parameters_gamma),fmt='%03.2f') 
        print 'res points fitted curve:',fittedResponseCurve 
        print 'res param gamma curve:', parameters_gamma 

    if selectExperiment_5:
        print 'exp 5 analysis'
        dataUser = np.loadtxt(dirToSaveResults+prefixName+'val_selected_by_user.txt')
        dataUser = np.vstack([vecSearchLevel, dataUser])
        function_display_RC_by_user(dataUser)

    if selectExperiment_1 & selectExperiment_2:
        funPlotTwoResponseCurves(vecLevel, responseCurve, vecSearchLevel, responseCurve2, dirToSaveResults+prefixName+'_c12_1.png')
        #print 'RC 1: ',responseCurve
        #print 'RC 2: ',responseCurve2

    if selectExperiment_2 & selectExperiment_5:
        funPlotTwoResponseCurves(vecLevel, responseCurve2, vecSearchLevel, dataUser[1,:], dirToSaveResults+prefixName+'_c12_2.png')

    if selectExperiment_3:
        # The data have been saved when the experiment 3 has been selected.
        print 'hop la boum! it''ratio time!'
        dataRatioRGB_L  = np.loadtxt(dirToSaveResults+prefixName+'_diffRatioRGB_L.txt')
        dataRatioRGB_Y  = np.loadtxt(dirToSaveResults+prefixName+'_diffRatioRGB_Y.txt')
        dataResTarget_Y = np.loadtxt(dirToSaveResults+prefixName+'_diffResTarget_Y.txt')        
        dataResBasis_Y  = np.loadtxt(dirToSaveResults+prefixName+'_diffResBasis_Y.txt')        

        np.savetxt(dirToSaveResults+prefixName+'_vecLevel.txt', (vecLevel),fmt='%03.2f')        
        np.savetxt(dirToSaveResults+prefixName+'_vecSearchLevelExp3.txt', (vecSearchLevelExp3),fmt='%03.2f')        

        ratioRGB1 = funComputeRatioFromExperiment(dataRatioRGB_Y, dataRatioRGB_L, vecLevelBasis, vecLevelTarget, dataResTarget_Y, dataResBasis_Y, 'config 1', dirToSaveResults+prefixName+'_ratio_Y1.png')

        dataRatioRGB_L2  = np.loadtxt(dirToSaveResults+prefixName+'_diffRatioRGB_L2.txt')
        dataRatioRGB_Y2  = np.loadtxt(dirToSaveResults+prefixName+'_diffRatioRGB_Y2.txt')
        dataResTarget_Y2 = np.loadtxt(dirToSaveResults+prefixName+'_diffResTarget_Y2.txt')        
        dataResBasis_Y2  = np.loadtxt(dirToSaveResults+prefixName+'_diffResBasis_Y2.txt')        

        ratioRGB2 = funComputeRatioFromExperiment(dataRatioRGB_Y2, dataRatioRGB_L2, vecLevelBasis2, vecLevelTarget2, dataResTarget_Y2, dataResBasis_Y2, 'config 2', dirToSaveResults+prefixName+'_ratio_Y2.png')

        #print dataRatioRGB_Y
        #print dataRatioRGB_Y2        

    if selectExperiment_2 & selectExperiment_3:
        # then we can recompute or compute the true ratio, the holy graal of this project
        Y_R_rel, Y_G_rel, Y_B_rel, ratio_GR, ratio_GB = funGiveMeTheRatio(ratioRGB1, vecLevelBasis[0], parameters_gamma)
        dataRatio = np.hstack([Y_R_rel, Y_G_rel, Y_B_rel, ratio_GR, ratio_GB])
        np.savetxt(dirToSaveResults+prefixName+'_final_ratio_1.txt', (dataRatio),fmt='%03.2f')        
        Y_R_rel, Y_G_rel, Y_B_rel, ratio_GR, ratio_GB = funGiveMeTheRatioBySpectro(ratioRGB1, vecLevelBasis[0], dataRCbyStellar)


        Y_R_rel, Y_G_rel, Y_B_rel, ratio_GR, ratio_GB = funGiveMeTheRatio(ratioRGB2, vecLevelBasis2[0], parameters_gamma)
        dataRatio2 = np.hstack([Y_R_rel, Y_G_rel, Y_B_rel, ratio_GR, ratio_GB])
        np.savetxt(dirToSaveResults+prefixName+'_final_ratio_2.txt', (dataRatio),fmt='%03.2f')        
        Y_R_rel, Y_G_rel, Y_B_rel, ratio_GR, ratio_GB = funGiveMeTheRatioBySpectro(ratioRGB2, vecLevelBasis[0], dataRCbyStellar)

    plt.show()

main()  
print 'well done, you reach the biggest achievement of your day, or maybe the second, not bad.'
