'''
This code is doing one thing: to measure the response curve of a display using a webcam.
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

configTest = False
configMeasurement = True
number_camera = 0
max_number_frame_to_wait_between_measurement = 50
max_number_frame_to_keep  = 25
tabOscillographeDifferences = np.zeros((4,max_number_frame_to_keep))
max_number_point_to_show = 3
mid_level = 128
sizeTilePatchHT = 256  # parameter for the HT patches with halftoning by mask
dirToSaveResults = './'
workDir = os.getcwd()

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
    global selectExperiment_1
    global selectExperiment_2

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
    np.savetxt(dirToSaveResults+prefixName+'_vecSearchLevel.txt', (vecSearchLevel),fmt='%03.2f') 
    np.savetxt(dirToSaveResults+prefixName+'_vecLevel.txt', (vecLevel),fmt='%03.2f')        
    
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
            tabAnsweruser = function_get_response_curve_from_human(widthFrame, heightFrame)
            print tabAnsweruser
            print 'You can start over if you want. Just if you want.'
            
            # save all the measurement as text
            np.savetxt('./'+prefixName+'val_selected_by_user.txt', (tabAnsweruser),fmt='%03.2f')

        if k_pressed == ord('q'):
            print 'Il faut sortir maintenant, faut pas rester ici.'
            break

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
            cv.SaveImage('./'+testChartName, imgT)
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
        cv.SaveImage('./'+testChartName, imgT)
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

def funComputeResponseCurveFromMeasurement(tabMeasurement, vectorLevel, vectorSearchLevel):
    '''
    The function does some processing like interpolation on the obtained curves, this to
    find the local minima corresponding to the search value for a given digital input.

    It also display curves as figure and save them.

    Args:
        tabMeasurement [floats]    : some intensity measurement Y or L 
        vectorLevel [floats]       : n points for which we did the measurement
        vectorSearchLevel [floats] : m > n points when we searched/measured the RC
    
    Ouptut:
        responsecurve [float]      : n points corresponding to the RC

    '''
    
    # Some interpolation for the Y measurements of target and basis
    interpInput, interpM = funInterpolationSingleCurve(np.vstack([vectorSearchLevel,tabMeasurement]))
    
    # and we normalize the bazar
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

def funComputeFittedResponseCurve(responseCurve, vec_meas_curve):#, data_stellar):
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
    #print 'Hammer time----------------------------------------'
    #print 'ICI REGARDE'
    #print 'curve fit',estimated_param, np.shape(estimated_param)
    #print 'lseastsq', plsq[0], np.shape(plsq[0])
    #print 'Hammer time----------------------------------------'
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
    #print np.shape(data_stellar)
    #plt.plot(data_stellar[:,0], data_stellar[:,1],'.-k',label='Spectro')

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
    print 'Now we process the data'
    
    # COMPUTE the response curve (RC) using Method 1:
    dataUser = np.loadtxt(dirToSaveResults+prefixName+'val_selected_by_user.txt')
    dataUser = np.vstack([vecSearchLevel, dataUser])
    function_display_RC_by_user(dataUser)
    plt.savefig('RCbyUserJIB.png')
    plt.show()


main()  
print 'well done, you reach the biggest achievement of your day, or maybe the second, not bad.'
  