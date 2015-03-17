
'''
Created on 21.01.2013
Here we position the camera to do measurement once the configuratin is
well enough adjusted. 
@author: gerjer
'''

import sys
import cv2.cv as cv
import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import Image
import time
from colorConversion import *
from colorDifferences import *

def function_initialize_camera(select_camera):
    ''' Initialize camera parameters for capturing video signal 

    Args: 
        select_camera (float or [floats]): a number 0, 1 depending of your camera 
        number, 0 corresponding often to integrated laptop webcam

    Output:
        camera (camera object)
    '''
    
    camera = cv.CreateCameraCapture(select_camera)

    if not camera:
        print "Could not open webcam!"
        sys.exit(1)
    
    cv.SetCaptureProperty(camera, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cv.SetCaptureProperty(camera, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    frame = cv.QueryFrame(camera)
    
    if frame is not None:
        w = frame.width
        h = frame.height
    
    time.sleep(1)
    
    return camera

def function_capture_image(file_name_frame, frame_to_be_saved, dir_to_save_image):
    ''' The function saves the last image taken by the camera.
    Args:
        file_name_frame (str)
        frame_to_be_saved (image
        dir_to_save_image (str)            
    '''

    cv.SaveImage(dir_to_save_image+file_name_frame, frame_to_be_saved)



def funDiffPatches(subA, subB):
    '''
    Thef function take two rectangle or images and compute the average difference between them
    using a simple distance formula.
    '''
    subMatA = np.asarray(subA)
    subMatB = np.asarray(subB)
    subAvgA = np.zeros(3)
    subAvgB = np.zeros(3)
    for i in np.arange(0,3,1):
        #print i
        subAvgA[i] = np.average(subMatA[:,:,i])
        subAvgB[i] = np.average(subMatB[:,:,i])
    diffRectangleAB = np.sqrt(np.sum((subAvgA-subAvgB)*(subAvgA-subAvgB)))
    return diffRectangleAB, subAvgA, subAvgB

def function_display_live_information(img, widthImg, heightImg, rec1, rec2):
    ''' The function take an image as input an computes the differences between
    two subregions. It returns the differences in RGB, XYZ, LAB, Y, L and 
    display on the image how the differnces evolve.

    Args:
        img: image
        widthImg (float [floats]): image width 
        heightImg (float [floats]): image height
        rec1 (array float [floats]): rectangle coords x y width height
        rec2 (array float [floats]): rectangle coords x y width height

    Ouptut:
        img: image with information overlay on the top of it. 
        diffL (float [floats])  : L intensity difference between the two rectangle areas
        diffLab (float [floats]): Lab difference between the two rectangle areas
        Lab1  (float [floats])  : L a b values left rectangle
        Lab2 (float [floats])   : L a b values right rectangle
        diffY (float [floats])  : Y intensity difference between the two rectangle areas
        XYZ1  (float [floats])  : X Y Z values left rectangle
        XYZ2 (float [floats])   : X Y Z values right rectangle
        diffRGB (float [floats]): RGB difference between the two rectangle areas
    '''

    # conversion to Lab
    imFilter = cv.CreateImage((widthImg,heightImg), cv.IPL_DEPTH_8U,3)
    cv.CvtColor(img, imFilter,cv.CV_BGR2Lab)
    imFilter = img
       
    # compute average color of each rectangle
    sub1 = cv.GetSubRect(img, (rec1[0], rec1[1], rec1[2], rec1[3]))
    sub2 = cv.GetSubRect(img, (rec2[0], rec2[1], rec2[2], rec2[3]))
                
    # compute differences in RGB
    diffRGB, subAvgRGB1, subAvgRGB2 = funDiffPatches(sub1, sub2)
            
    # conversion to XYZ and Lab
    RGB1 = np.asarray(subAvgRGB1)
    RGB1 = np.reshape(RGB1,(3,-1))
    XYZ1 = conversion_RGB_to_XYZ(RGB1)
    RGB2 = np.asarray(subAvgRGB2)
    RGB2 = np.reshape(RGB2,(3,-1))
    XYZ2 = conversion_RGB_to_XYZ(RGB2)
    Lab1 = conversion_XYZ_to_Lab(XYZ1, 'D50_31')
    Lab2 = conversion_XYZ_to_Lab(XYZ2, 'D50_31')
    diffY   = np.sqrt((XYZ1[1]-XYZ2[1]) * (XYZ1[1]-XYZ2[1]))
    diffLab = fun_difference_CIELab76(Lab1, Lab2)
    diffL   = np.sqrt((Lab1[0]-Lab2[0]) * (Lab1[0]-Lab2[0]))

    # prepar the font
    font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 8) #Creates a font
    
    # display animation to show RGB the difference between patches
    #img = funAddRectangleForErrorVisualization(diffRGB/255, 320, 180, 190, colorRectangleAndText,img)
    diffInText = "%3.2f" % diffRGB
    diffInText = 'RGB: '+diffInText
    colorRectangleAndText = (255,20,147)
    img = funAddRectangleForErrorVisualizationAndText(diffRGB/255, 320, 180, 190, diffInText, font, colorRectangleAndText, img)

    # display animation to show Y difference after conversion RGB -> YXZ
    diffInText = "%3.2f" % diffY
    diffInText = 'Y: '+diffInText
    colorRectangleAndText = (221,160,221)
    img = funAddRectangleForErrorVisualizationAndText(diffY/100, 320, 120, 130, diffInText, font, colorRectangleAndText, img)
        
    # display animation to show Lab the difference between patches
    diffInText = "%3.2f" % diffLab
    diffInText = 'Lab: '+diffInText
    colorRectangleAndText = (186, 85, 211)
    img = funAddRectangleForErrorVisualizationAndText(diffLab/100, 320 ,160, 170, diffInText, font, colorRectangleAndText, img)

    # display animation to show "L" brighness difference between patches
    diffInText = "%3.2f" % diffL
    diffInText = 'L: '+diffInText
    colorRectangleAndText = (148,0,211)
    img = funAddRectangleForErrorVisualizationAndText(diffL/100, 320 , 140 ,150, diffInText, font, colorRectangleAndText,img)
        
    # display information on image
    cv.Rectangle(img, (rec1[0], rec1[1]), (rec1[0]+rec1[2], rec1[1]+rec1[3]), (0,0,255))
    cv.Rectangle(img, (rec2[0], rec2[1]), (rec2[0]+rec2[2], rec2[1]+rec2[3]), (0,0,255))
    cv.Line(img, (320,10), (320, 470), (255, 255, 255))

    # display CIE ab diagram
    img = funAddCircleCIEabVisualization((320, 240), 200, Lab1, Lab2, RGB1, RGB2, diffLab, img)

    return img, diffL, diffLab, Lab1, Lab2, diffY, XYZ1, XYZ2, diffRGB

def function_display_oscilloscope(image, width_frame, height_frame, tab_oscillographe_differences):
    '''The function takes a series of point as input and draw a line. It gives
    a kind of oscilloscope visualization.

    Args:
        image
        width_frame (float [floats]): image width
        height_frame (float [floats]): image height
        tab_oscillographe_differences (float [floats]): table of differencs of size 4 x n

    Output:
        image, same as input but with information on the top of it.

    '''

    # create the x points:
    x  = np.linspace(0,width_frame,np.shape(tab_oscillographe_differences)[1])
    y1 = height_frame - tab_oscillographe_differences[0,:]
    y2 = height_frame - tab_oscillographe_differences[1,:]
    y3 = height_frame - tab_oscillographe_differences[2,:]
    y4 = height_frame - tab_oscillographe_differences[3,:]
    for ii in np.arange(1,np.shape(tab_oscillographe_differences)[1]):
        cv.Line(image, (int(x[ii-1]), int(y1[ii-1])), (int(x[ii]), int(y1[ii])), (128,0,211))
        cv.Line(image, (int(x[ii-1]), int(y2[ii-1])), (int(x[ii]), int(y2[ii])), (186,85,211))
        cv.Line(image, (int(x[ii-1]), int(y3[ii-1])), (int(x[ii]), int(y3[ii])), (221,160,221))
        cv.Line(image, (int(x[ii-1]), int(y4[ii-1])), (int(x[ii]), int(y4[ii])), (255,20,147))
    return image

def funAddRectangleForErrorVisualization(diff, posx, posy1, posy2, colorObject, image):
    '''The function takes a difference value as entree and displays it as a
    rectangle or horizontal error bar on the frame , frame also given as input 
    to the function.

    Args:
        diff (float [floats]) : difference value
        posx (floaf [floats]) : parameter for the x position of the error bar
        posy1 (floaf [floats]): parameter for the left side y position of the error bar
        posy2 (floaf [floats]): parameter for the right side y position of the error bar
        image: an image

    Output:
        image with the error bar overlaid

    '''
    diff = np.int32(np.round(diff * 120))
    cv.Rectangle(image, (posx-diff,posy1), (posx,posy2), colorObject)
    cv.Rectangle(image, (posx,posy1), (posx+diff,posy2), colorObject)
    
    return image

def funAddRectangleForErrorVisualizationAndText(diff, posx, posy1, posy2, diffInText, font, colorObject, image):
    '''
    The function takes a difference value as entree and displays it as a rectangle
    on the frame , frame also given as input to the function.
    '''
    diff = np.int32(np.round(diff * 120))
    cv.Rectangle(image, (posx-diff,posy1), (posx,posy2), colorObject)
    cv.Rectangle(image, (posx,posy1), (posx+diff,posy2), colorObject)   
    
    # write the Delta in the middle of the line between the two points
    cv.PutText(image, diffInText, (posx+diff+2, posy2),font, colorObject) 

    return image

def funAddCircleCIEabVisualization(centerCircle, radiusCircle, Lab1, Lab2, rgb1, rgb2, differenceLab, image):
    #img = funAddCircleCIEabVisualization(380, 240, 100, Lab1, Lab2, img)
    '''
    The function takes as input the average Lab of each rectangle and display the
    ab information on a ab diagram. Also a line betwee the two points is plotted.
    '''
    # draw circle defining the ab plan
    #cv.Circle(image, centerCircle, radiusCircle, (0,0,255), thickness=5, lineType=8, shift=0)
    # draw line showing distance betweem the two rectangles
    centerCircle1 = ((Lab1[1]/128)*radiusCircle+320, (Lab1[2]/128)*radiusCircle+340)
    centerCircle2 = ((Lab2[1]/128)*radiusCircle+320, (Lab2[2]/128)*radiusCircle+340)
    cv.Circle(image, centerCircle1, 2, (int(rgb1[0]),int(rgb1[1]),int(rgb1[2])), thickness=int(Lab1[0]), lineType=8, shift=0)
    cv.Circle(image, centerCircle2, 2, (int(rgb2[0]),int(rgb2[1]),int(rgb2[2])), thickness=int(Lab2[0]), lineType=8, shift=0)
    
    cv.Line(image, centerCircle1, centerCircle2, (255,255, 255), thickness=4)

    return image

def funInterpolationRGBcurve(dataRGB):
    ''' The function, obviously, does some interpolation.

    Args: 
        dataRGB (array float [floats]): table of data of size 4 x N where
        dataRGB[0,:] are the x values
        dataRGB[1,:] are the corresponding difference for the red channel
        dataRGB[2,:] are the corresponding difference for the green channel
        dataRGB[3,:] are the corresponding difference for the blue channel

    Output:
        interpX (array float [floats]): new entree points for which we have 
        interpolation data.
        interpR (array float [floats]):  interpolated data for the R channel
        interpG (array float [floats]):  interpolated data for the G channel
        interpB (array float [floats]):  interpolated data for the B channel

    '''
    interpX = np.arange(dataRGB[0,0], dataRGB[0,-1])
    interpR = np.interp(interpX, dataRGB[0,:], dataRGB[1,:])
    interpG = np.interp(interpX, dataRGB[0,:], dataRGB[2,:]) # interpolation of the diff data
    interpB = np.interp(interpX, dataRGB[0,:], dataRGB[3,:])
    return interpX, interpR, interpG, interpB

def funInterpolationContinuousHalftonecurve(vecDigitalInput, dataContinuous, dataHalftoned):
    '''
    The function, obviously, does some interpolation.
    '''
    interpX = np.arange(vecDigitalInput[0], vecDigitalInput[-1]+1)
    #print np.shape(dataContinuous)
    #print np.shape(vecDigitalInput)
    #interpolationContinuous = np.interp(interpX, vecDigitalInput, dataContinuous)
    tck = interpolate.splrep(vecDigitalInput,dataContinuous,s=0)
    interpolationContinuous = interpolate.splev(interpX,tck,der=0)
    #interpolationHalftone   = np.interp(interpX, vecDigitalInput, dataHalftoned)
    tck = interpolate.splrep(vecDigitalInput,dataHalftoned,s=0)
    interpolationHalftone = interpolate.splev(interpX,tck,der=0)

    return interpX, interpolationContinuous, interpolationHalftone

def funInterpolationRatioDifferenceCurves(vecDigitalInput, dataRatioRGB):
    '''
    The function, obviously, does some interpolation, this time for the rati0 experiment.
    '''
    interpX = np.arange(vecDigitalInput[0], vecDigitalInput[-1])
    interpData = np.zeros((3,np.size(interpX)))
    for ii in np.arange(0,3):
        tck = interpolate.splrep(vecDigitalInput,dataRatioRGB[ii,:],s=0)
        interpData[ii,:] = interpolate.splev(interpX,tck,der=0)

    #interpData[0,:] = np.interp(interpX, vecDigitalInput, dataRationRGB[0,:])
    #interpData[1,:] = np.interp(interpX, vecDigitalInput, dataRationRGB[1,:])
    #interpData[2,:] = np.interp(interpX, vecDigitalInput, dataRationRGB[2,:])
    return interpX, interpData

def funInterpolationSingleCurve(data):
    '''
    The function, obviously, does some interpolation.
    data are only one channel (as opposite to the function above)
    but are of the shape data(x,y)
    '''
    #print np.shape(data)
    #print data
    interpX = np.arange(data[0,0], data[0,-1])
    interpY = np.zeros((np.shape(data)[0]-1,np.size(interpX)))
    for ii in np.arange(0,np.shape(data)[0]-1):
        #interpY[ii,:] = np.interp(interpX, data[0,:], data[ii+1,:])
        tck = interpolate.splrep(data[0,:],data[ii+1,:],s=0)
        #print np.shape(tck)
        interpY[ii,:] = interpolate.splev(interpX,tck,der=0)
    return interpX, interpY

def funGetSomeMinimum(interpR, interpG, interpB):
    ''' Function used by the funGetMinForEachChannel to find the minimum 
    in each interp entree.

    Args:
        interpR (array floats[floats]): vector of point
        interpG (array floats[floats]): vector of point
        interpB (array floats[floats]): vector of point

    Outputs:
        minR, minG, minB (float [floats]): min value for each entree
        indR, indG, indB (float [floats]): corresponding index value where the 
        minimun value is located in each entree.

    '''
    
    # get the intensity giving the minimum diff
    minR = np.min(interpR)
    minG = np.min(interpG) 
    minB = np.min(interpB)

    # get the inedx of minimumv
    indR = np.argmin(interpR)
    indG = np.argmin(interpG) 
    indB = np.argmin(interpB)

    return minR, minG, minB, indR, indG, indB

def funGetSomeMinimumSingleCurve(interpM):
    '''
    Function used by the funGetMinForEachChannel to find the minimum 
    in one single interp entree.

    In theory this function is giving me the response curve.
    '''
    print 'size interpM is ',np.shape(interpM)
    minM = np.zeros((np.shape(interpM)[0],1))
    indM = np.zeros((np.shape(interpM)[0],1))
    for ii in np.arange(0,np.shape(interpM)[0]):
        # get the intensity giving the minimum diff
        minM[ii] = np.min(interpM[ii,:])
        # get the inedx of minimumv
        indM[ii] = np.argmin(interpM[ii,:])
    return minM, indM

def funDoErrorDiffusion_FloydSteinberg(imageData):
    '''
    As it is written in the name of the function, this function does error error
    diffusion using the Floyd-Steinberg weights to diffuse the errors.
    '''
    imageDataHalftoned = np.zeros(np.shape(imageData))
    #print np.shape(imageData)
    matrice_weight = np.array([[0, 0, 7],[3, 5, 1]]) / 16.
    #print matrice_weight
    for ii in np.arange(1,np.shape(imageDataHalftoned)[0]-1):
        for jj in np.arange(0,np.shape(imageData)[1]-1):
            if imageData[ii,jj] >= 0.5:
                imageDataHalftoned[ii,jj] = 1
                # compute the difference
                pixelDiff = imageData[ii,jj] - 1
            else:
                imageDataHalftoned[ii,jj] = 0
                # compute the difference
                pixelDiff = imageData[ii,jj] - 0
            
            # spread the error
            imageData[ii+1,jj]   = pixelDiff * matrice_weight[0,2] + imageData[ii+1,jj]  # first line
            imageData[ii-1,jj+1] = pixelDiff * matrice_weight[1,0] + imageData[ii-1,jj+1] # second line
            imageData[ii,  jj+1] = pixelDiff * matrice_weight[1,1] + imageData[ii  ,jj+1] # second line
            imageData[ii+1,jj+1] = pixelDiff * matrice_weight[1,2] + imageData[ii+1,jj+1] # second line

    return imageDataHalftoned

def funDoErrorDiffusion_JarvisAndAll(imageData):
    '''
    As it is written in the name of the function, this function does error error
    diffusion using the Jarvis and All weights to diffuse the errors.
    '''
    imageDataHalftoned = np.zeros(np.shape(imageData))
    matrice_weight = np.array([[0., 0., 0., 7., 5.],[3., 5., 7., 5., 3.],[1., 3., 5., 3., 1.]]) / 48.
    for ii in np.arange(2,np.shape(imageDataHalftoned)[0]-2):
        for jj in np.arange(0,np.shape(imageData)[1]-2):
            if imageData[ii,jj] >= 0.5:
                imageDataHalftoned[ii,jj] = 1
                # compute the difference
                pixelDiff = imageData[ii,jj] - 1
            else:
                imageDataHalftoned[ii,jj] = 0
                # compute the difference
                pixelDiff = imageData[ii,jj] - 0
            
            # spread the error
            imageData[ii+1,jj]   = pixelDiff * matrice_weight[0,3] + imageData[ii+1,jj]   # first line
            imageData[ii+2,jj]   = pixelDiff * matrice_weight[0,4] + imageData[ii+2,jj]
            imageData[ii-2,jj+1] = pixelDiff * matrice_weight[1,0] + imageData[ii-2,jj+1]  # second line
            imageData[ii-1,jj+1] = pixelDiff * matrice_weight[1,1] + imageData[ii-1,jj+1]
            imageData[ii,  jj+1] = pixelDiff * matrice_weight[1,2] + imageData[ii  ,jj+1]
            imageData[ii+1,jj+1] = pixelDiff * matrice_weight[1,3] + imageData[ii+1,jj+1]
            imageData[ii+2,jj+1] = pixelDiff * matrice_weight[1,4] + imageData[ii+2,jj+1]

            imageData[ii-2,jj+2] = pixelDiff * matrice_weight[2,0] + imageData[ii-2,jj+2]  # third line
            imageData[ii-1,jj+2] = pixelDiff * matrice_weight[2,1] + imageData[ii-1,jj+2]
            imageData[ii,  jj+2] = pixelDiff * matrice_weight[2,2] + imageData[ii  ,jj+2]
            imageData[ii+1,jj+2] = pixelDiff * matrice_weight[2,3] + imageData[ii+1,jj+2]
            imageData[ii+2,jj+2] = pixelDiff * matrice_weight[2,4] + imageData[ii+2,jj+2]

    ## flip the image to get the best distribution close to the border
    ##imageDataHalftoned = np.fliplr(imageDataHalftoned)
    return imageDataHalftoned


def imCreateTestchartPatchBase(levelPatch, levelBase):
    '''
    The function takes two input values as input such that levelPatch < levelBase.
    The resulting test chart has a fixed size-
    
    Args:
        levelPatch        : float 
        levelBase  (array): float 1 x 3

    out:
        imagePatchBase    : array corresponding to image size n x m x 3
    '''
    
    width = 1024    
    height = 720
    imagePatchBase = np.zeros((height, width,3), np.uint8)
     
    imagePatchBase[:,0:0.5*width,:] = levelPatch
    imagePatchBase[:,0.5*width:width,0] = levelBase[0]
    imagePatchBase[:,0.5*width:width,1] = levelBase[1]
    imagePatchBase[:,0.5*width:width,2] = levelBase[2]
    
    return imagePatchBase


def imCreateTestChartSingleColor(levelRGB):
    '''
    The function takes an RGB value and create an image of the same value.
    '''
    width = 1024
    height = 720
    imagePatchSingleColor = np.zeros((height, width,3), np.uint8)
         
    imagePatchSingleColor[:,:,0] = levelRGB[0]
    imagePatchSingleColor[:,:,1] = levelRGB[1]
    imagePatchSingleColor[:,:,2] = levelRGB[2]
    
    return imagePatchSingleColor

def funDoHalftoningByMaskLinear(im):
    '''
    The function halfotned the image using the mask technique
    The fitler mask are created everytime the function is called.
    '''
    size_mask = 16
    size_image = np.shape(im)

    # check if the image size is a multiple of the mask size
    rest_division_image_size = np.zeros((2,2))
    rest_division_image_size[0,:] = divmod(size_image[0], size_mask)
    rest_division_image_size[1,:] = divmod(size_image[1],size_mask)
    
    # create the mask random
    data_mask = np.arange(0,256)
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 255.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # creat the image mask 
    size_data_mask_image_factor = np.zeros(2)
    if rest_division_image_size[0,1] != 0:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0] + 1
    else:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0]

    if rest_division_image_size[1,1] != 0:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0] + 1
    else:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0]
    image_data_mask = np.tile(data_mask,(size_data_mask_image_factor[0], size_data_mask_image_factor[1]))
    size_data_mask_image = np.shape(image_data_mask)

    # add border to the image before halftoning
    image_with_border = np.zeros((size_data_mask_image[0], size_data_mask_image[1]))
    image_with_border[0:size_image[0], 0:size_image[1]] = im

    # do the halftoning
    image_halftoned = image_with_border <= image_data_mask
    image_halftoned = image_halftoned.astype(float)

    # and we remove the added border:
    image_halftoned = image_halftoned[0:size_image[0],0:size_image[1]]
    
    return image_halftoned

def funDoHalftoningByMaskRandom(im):
    '''
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, only one mask is created.
    '''
    imHalfoned = np.zeros(np.shape(im))
    size_mask = 16
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    #data_mask = np.resize(data_mask,(16,16))
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 255.
    print data_mask, data_mask.dtype
    data_mask = np.resize(data_mask,(size_mask,size_mask))
    print data_mask, data_mask.dtype
    # apply the mask on the image
    vec_X = np.arange(0,np.shape(im)[0],size_mask)
    vec_Y = np.arange(0,np.shape(im)[1],size_mask)
    for ii in vec_X[0:-1]:
        for jj in vec_Y[0:-1]:
            # select the block in the image
            data_block = im[ii:ii+size_mask, jj:jj+size_mask]
            # thresholding
            data_block[data_block <= data_mask] = 0
            data_block[data_block > data_mask] = 1
            imHalfoned[ii:ii+size_mask, jj:jj+size_mask] = data_block
            print data_block
    return imHalfoned

def funDoHalftoningByMaskRandomRGB(imRGB):
    '''
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, only one mask is created.
    '''
    imHalfonedRGB = np.zeros(np.shape(imRGB))
    size_mask = 16
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    data_mask = np.resize(data_mask,(16,16))
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 256.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # apply the mask on the image
    vec_X = np.arange(0,np.shape(imRGB)[0],size_mask)
    vec_Y = np.arange(0,np.shape(imRGB)[1],size_mask)
    for kk in np.arange(0,3):
        for ii in vec_X[0:-1]:
            for jj in vec_Y[0:-1]:
                # select the block in the image
                data_block = imRGB[ii:ii+size_mask, jj:jj+size_mask,kk]
                # thresholding
                data_block[data_block <= data_mask] = 0
                data_block[data_block > data_mask] = 1
                imHalfonedRGB[ii:ii+size_mask, jj:jj+size_mask,kk] = data_block
    return imHalfonedRGB

def funDoHalftoningByMaskFullRandom(im):
    '''
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, for every processed block 
    a new mask is created. But this time we try to be smart.
    '''
    
    size_mask = 16
    size_image = np.shape(im)

    # check if the image size is a multiple of the mask size
    rest_division_image_size = np.zeros((2,2))
    rest_division_image_size[0,:] = divmod(size_image[0], size_mask)
    rest_division_image_size[1,:] = divmod(size_image[1], size_mask)
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)

    data_mask = data_mask  / 256.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # create the image mask 
    size_data_mask_image_factor = np.zeros(2)
    if rest_division_image_size[0,1] != 0:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0] + 1
    else:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0]

    if rest_division_image_size[1,1] != 0:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0] + 1
    else:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0]
    
    image_data_mask = np.tile(data_mask,(size_data_mask_image_factor[0], size_data_mask_image_factor[1]))
    size_data_mask_image = np.shape(image_data_mask)

    # add border to the image before halftoning
    image_with_border = np.zeros((size_data_mask_image[0], size_data_mask_image[1]))
    image_with_border[0:size_image[0], 0:size_image[1]] = im

    # do the halftoning
    image_halftoned = np.zeros((np.shape(image_with_border)[0],np.shape(image_with_border)[1]))
    image_halftoned[image_with_border > image_data_mask] = 1
    #image_halftoned = image_halftoned.astype(float)

    # and we remove the added border:
    image_halftoned = image_halftoned[0:size_image[0],0:size_image[1]]
    
    return image_halftoned

def funDoHalftoningByMaskFullRandomRGB(imRGB):
    '''
    The same as above but in color
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, for every processed block 
    a new mask is created. But this time we try to be smart.
    '''
    
    size_mask = 16
    size_image = np.shape(imRGB)

    # check if the image size is a multiple of the mask size
    rest_division_image_size = np.zeros((2,2))
    rest_division_image_size[0,:] = divmod(size_image[0], size_mask)
    rest_division_image_size[1,:] = divmod(size_image[1], size_mask)
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 255.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # create the image mask 
    size_data_mask_image_factor = np.zeros(2)
    if rest_division_image_size[0,1] != 0:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0] + 1
    else:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0]

    if rest_division_image_size[1,1] != 0:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0] + 1
    else:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0]
    
    # create only one channel then expand
    image_data_mask_one_channel = np.tile(data_mask,(size_data_mask_image_factor[0], size_data_mask_image_factor[1]))
    size_data_mask_image_one_channel = np.shape(image_data_mask_one_channel)
    image_data_mask = np.zeros((size_data_mask_image_one_channel[0], size_data_mask_image_one_channel[1], 3))
    print size_data_mask_image_one_channel
    image_data_mask[:,:,0] = image_data_mask_one_channel
    image_data_mask[:,:,1] = image_data_mask_one_channel
    image_data_mask[:,:,2] = image_data_mask_one_channel

    # add border to the image before halftoning
    image_with_border = np.zeros((size_data_mask_image_one_channel[0], size_data_mask_image_one_channel[1], 3))
    image_with_border[0:size_image[0], 0:size_image[1],:] = imRGB

    # do the halftoning
    image_halftoned = image_with_border <= image_data_mask
    image_halftoned = image_halftoned.astype(float)

    # and we remove the added border:
    image_halftoned = image_halftoned[0:size_image[0],0:size_image[1],:]
    
    return image_halftoned
    

def function_display_RC_by_user(data_by_User):
    ''' The function display the response curve from level selected
    by a user.

    Args:
        data_by_User: array (float [floats]) of 2 x n where the first 
        line are the digital step value of the ramp in continuous tone.

    Output:
        None so far but it will come one day

    '''

    fig = plt.figure()
    plt.plot(data_by_User[0,:], data_by_User[1,:] / 255.,'bo-', label='Human')
    plt.plot(data_by_User[0,:], data_by_User[0,:] / 255.,'k-', label='linear')
    plt.plot(data_by_User[0,:], np.power(data_by_User[0,:]/255., 2.2), 'k:', label='gamma')
    plt.xlabel('digital input')
    plt.ylabel('digital Ding')
    plt.title('Response Curve by Human Observer')
    plt.legend(loc=2)
    plt.xlim(0,255)
    plt.ylim(0,1)
    plt.draw()

def initCamera(selectCamera):
    ''' 
    Initialize camera parameters
    '''
    camera = cv.CreateCameraCapture(selectCamera)
    if not camera:
        print "Could not open webcam!"
        sys.exit(1)
    cv.SetCaptureProperty(camera, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cv.SetCaptureProperty(camera, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    frame = cv.QueryFrame(camera)
    if frame is not None:
        w = frame.width
        h = frame.height
    #    print "%d %d"%(w, h)
    time.sleep(1)
    return camera

def funCaptureImage(fileNameFrame,frameToSaved,dirToSaveImage):
    '''
    The function saves the last image taken by the camera.
    '''
    cv.SaveImage(dirToSaveImage+fileNameFrame, frameToSaved)
    #print "one picture saved, good job!"

def imCreateTestchartContinuousAndHalftoned(levelContinuous, levelHalftoned, sizeTileHalftone, background=128):
    '''
    Here I create the image, one side continuous and the other halftoned.
    The sizeTileHalftone parameter is used to play with the size of the point in 
    the HT patch. sizeTileHalftone = 16 is the minimum.

    Actually it can be smaller than 16, but there will be a problem because
    of the Ht by mask, to have a minimum size of 16 ensures that 256 levels 
    available.

    Input:
        - levelContinuous float (floats) : level of the left part of the testchart
        - levelHalftoned float (floats)  : levle of the right part of the testchart
        - sizeTileHalftone float (floats): how big are the tile for the halftone part
        - background float (float)       : background level to get a uniform  background around
                                           the part to be measured.
    Output:
        - imageTestchartContinuousHalfoned np table.
    '''
    width = 1024
    height = 720
    imageTestchartContinuousHalfoned = np.zeros((height, width), np.uint8)
    imageTestchartContinuousHalfoned[0:104,:] = 0
    imageTestchartContinuousHalfoned[617:height,:] = 0
    
    im1 = np.ones((512,512),np.uint8)*levelContinuous # continous part of the testchart
    
    #-- method Error Diffusion
    im2 = np.ones((sizeTileHalftone,sizeTileHalftone))* (levelHalftoned / 255.)
    #im2 = funDoErrorDiffusion_FloydSteinberg(im2)
    #im2 = funDoErrorDiffusion_JarvisAndAll(im2)
    im2 = funDoHalftoningByMaskFullRandom(im2)
    #im2 = funDoHalftoningByMaskLinear(im2)
    im_src = cv.fromarray(im2*255)
    im_dst = cv.CreateMat(512,512, cv.CV_64FC1)
    cv.Resize(im_src, im_dst, interpolation=cv.CV_INTER_NN)
    im22 = np.asarray(im_dst)
    
    # and finally
    imageTestchartContinuousHalfoned[104:616,0:512] = im1
    imageTestchartContinuousHalfoned[104:616,512:width] = im22

    # and now we set the part around the part to be measured to the background value
    if background != 0:
        # left and right
        imageTestchartContinuousHalfoned[:, 0:280]    =  np.ones((720,280),np.uint8)*background
        imageTestchartContinuousHalfoned[:, 744:1024] =  np.ones((720,280),np.uint8)*background
        # top and bottom
        imageTestchartContinuousHalfoned[0:180,:]      =  np.ones((180,1024),np.uint8)*background
        imageTestchartContinuousHalfoned[540:720,:] =  np.ones((180,1024),np.uint8)*background


    return imageTestchartContinuousHalfoned

def funDiffPatches(subA, subB):
    '''
    Thef function take two rectangle or images and compute the average difference between them
    using a simple distance formula.
    '''
    subMatA = np.asarray(subA)
    subMatB = np.asarray(subB)
    subAvgA = np.zeros(3)
    subAvgB = np.zeros(3)
    for i in np.arange(0,3,1):
        #print i
        subAvgA[i] = np.average(subMatA[:,:,i])
        subAvgB[i] = np.average(subMatB[:,:,i])
    diffRectangleAB = np.sqrt(np.sum((subAvgA-subAvgB)*(subAvgA-subAvgB)))
    return diffRectangleAB, subAvgA, subAvgB

def funDisplayLiveInformation(img, widthImg, heightImg, rec1, rec2):
    '''
    The function take an image as input an computes the differences between two subregions.
    It returns the differences in RGB, XYZ, LAB, Y, L and display on the image how the
    differnces evolve.
    To Do:
    - to add as input the coordinats of the subregions in percentage maybe???
    '''
    # conversion to Lab
    imFilter = cv.CreateImage((widthImg,heightImg), cv.IPL_DEPTH_8U,3)
    cv.CvtColor(img, imFilter,cv.CV_BGR2Lab)
    imFilter = img
       
    # compute average color of each rectangle
    sub1 = cv.GetSubRect(img, (rec1[0], rec1[1], rec1[2], rec1[3]))
    sub2 = cv.GetSubRect(img, (rec2[0], rec2[1], rec2[2], rec2[3]))
                
    # compute differences in RGB
    diffRGB, subAvgRGB1, subAvgRGB2 = funDiffPatches(sub1, sub2)
            
    # conversion to XYZ and Lab
    RGB1 = np.asarray(subAvgRGB1)
    RGB1 = np.reshape(RGB1,(3,-1))
    XYZ1 = conversion_RGB_to_XYZ(RGB1)
    RGB2 = np.asarray(subAvgRGB2)
    RGB2 = np.reshape(RGB2,(3,-1))
    XYZ2 = conversion_RGB_to_XYZ(RGB2)
    Lab1 = conversion_XYZ_to_Lab(XYZ1, 'D50_31')
    Lab2 = conversion_XYZ_to_Lab(XYZ2, 'D50_31')
    diffY   = np.sqrt((XYZ1[1]-XYZ2[1]) * (XYZ1[1]-XYZ2[1]))
    diffLab = fun_difference_CIELab76(Lab1, Lab2)
    diffL   = np.sqrt((Lab1[0]-Lab2[0]) * (Lab1[0]-Lab2[0]))

    # prepar the font
    font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 8) #Creates a font
    
    # display animation to show RGB the difference between patches
    #img = funAddRectangleForErrorVisualization(diffRGB/255, 320, 180, 190, colorRectangleAndText,img)
    diffInText = "%3.2f" % diffRGB
    diffInText = 'RGB: '+diffInText
    colorRectangleAndText = (255,20,147)
    img = funAddRectangleForErrorVisualizationAndText(diffRGB/255, 320, 180, 190, diffInText, font, colorRectangleAndText, img)

    # display animation to show Y difference after conversion RGB -> YXZ
    diffInText = "%3.2f" % diffY
    diffInText = 'Y: '+diffInText
    colorRectangleAndText = (221,160,221)
    img = funAddRectangleForErrorVisualizationAndText(diffY/100, 320, 120, 130, diffInText, font, colorRectangleAndText, img)
        
    # display animation to show Lab the difference between patches
    diffInText = "%3.2f" % diffLab
    diffInText = 'Lab: '+diffInText
    colorRectangleAndText = (186, 85, 211)
    img = funAddRectangleForErrorVisualizationAndText(diffLab/100, 320 ,160, 170, diffInText, font, colorRectangleAndText, img)

    # display animation to show "L" brighness difference between patches
    diffInText = "%3.2f" % diffL
    diffInText = 'L: '+diffInText
    colorRectangleAndText = (148,0,211)
    img = funAddRectangleForErrorVisualizationAndText(diffL/100, 320 , 140 ,150, diffInText, font, colorRectangleAndText,img)
        
    # display information on image
    cv.Rectangle(img, (rec1[0], rec1[1]), (rec1[0]+rec1[2], rec1[1]+rec1[3]), (0,0,255))
    cv.Rectangle(img, (rec2[0], rec2[1]), (rec2[0]+rec2[2], rec2[1]+rec2[3]), (0,0,255))
    cv.Line(img, (320,10), (320, 470), (255, 255, 255))

    # display CIE ab diagram
    img = funAddCircleCIEabVisualization((320, 240), 200, Lab1, Lab2, RGB1, RGB2, diffLab, img)

    return img, diffL, diffLab, Lab1, Lab2, diffY, XYZ1, XYZ2, diffRGB

def funDisplayOscilliscope(img, widthFrame, heightFrame, tabOscillographeDifferences):
    '''
    The function takes a series of point as input and draw a line.
    '''
    # create the x points:
    xPoints  = np.linspace(0,widthFrame,np.shape(tabOscillographeDifferences)[1])
    yPoints1 = heightFrame - tabOscillographeDifferences [0,:]
    yPoints2 = heightFrame - tabOscillographeDifferences [1,:]
    yPoints3 = heightFrame - tabOscillographeDifferences [2,:]
    yPoints4 = heightFrame - tabOscillographeDifferences [3,:]
    for ii in np.arange(1,np.shape(tabOscillographeDifferences)[1]  ):
        cv.Line(img, (int(xPoints[ii-1]), int(yPoints1[ii-1])), (int(xPoints[ii]), int(yPoints1[ii])), (128,0,211))
        cv.Line(img, (int(xPoints[ii-1]), int(yPoints2[ii-1])), (int(xPoints[ii]), int(yPoints2[ii])), (186,85,211))
        cv.Line(img, (int(xPoints[ii-1]), int(yPoints3[ii-1])), (int(xPoints[ii]), int(yPoints3[ii])), (221,160,221))
        cv.Line(img, (int(xPoints[ii-1]), int(yPoints4[ii-1])), (int(xPoints[ii]), int(yPoints4[ii])), (255,20,147))
    return img

def funAddRectangleForErrorVisualization(diff, posx, posy1, posy2, colorObject, image):
    '''
    The function takes a difference value as entree and displays it as a rectangle
    on the frame , frame also given as input to the function.
    '''
    diff = np.int32(np.round(diff * 120))
    cv.Rectangle(image, (posx-diff,posy1), (posx,posy2), colorObject)
    cv.Rectangle(image, (posx,posy1), (posx+diff,posy2), colorObject)
    
    return image

def funAddRectangleForErrorVisualizationAndText(diff, posx, posy1, posy2, diffInText, font, colorObject, image):
    '''
    The function takes a difference value as entree and displays it as a rectangle
    on the frame , frame also given as input to the function.
    '''
    diff = np.int32(np.round(diff * 120))
    cv.Rectangle(image, (posx-diff,posy1), (posx,posy2), colorObject)
    cv.Rectangle(image, (posx,posy1), (posx+diff,posy2), colorObject)   
    
    # write the Delta in the middle of the line between the two points
    cv.PutText(image, diffInText, (posx+diff+2, posy2),font, colorObject) 

    return image

def funAddCircleCIEabVisualization(centerCircle, radiusCircle, Lab1, Lab2, rgb1, rgb2, differenceLab, image):
    #img = funAddCircleCIEabVisualization(380, 240, 100, Lab1, Lab2, img)
    '''
    The function takes as input the average Lab of each rectangle and display the
    ab information on a ab diagram. Also a line betwee the two points is plotted.
    '''
    # draw circle defining the ab plan
    #cv.Circle(image, centerCircle, radiusCircle, (0,0,255), thickness=5, lineType=8, shift=0)
    # draw line showing distance betweem the two rectangles
    centerCircle1 = ((Lab1[1]/128)*radiusCircle+320, (Lab1[2]/128)*radiusCircle+340)
    centerCircle2 = ((Lab2[1]/128)*radiusCircle+320, (Lab2[2]/128)*radiusCircle+340)
    cv.Circle(image, centerCircle1, 2, (int(rgb1[0]),int(rgb1[1]),int(rgb1[2])), thickness=int(Lab1[0]), lineType=8, shift=0)
    cv.Circle(image, centerCircle2, 2, (int(rgb2[0]),int(rgb2[1]),int(rgb2[2])), thickness=int(Lab2[0]), lineType=8, shift=0)
    
    cv.Line(image, centerCircle1, centerCircle2, (255,255, 255), thickness=4)

    return image

def funInterpolationRGBcurve(dataRGB):
    '''
    The function, obviously, does some interpolation.
    '''
    interpX = np.arange(dataRGB[0,0], dataRGB[0,-1])
    interpR = np.interp(interpX, dataRGB[0,:], dataRGB[1,:])
    interpG = np.interp(interpX, dataRGB[0,:], dataRGB[2,:]) # interpolation of the diff data
    interpB = np.interp(interpX, dataRGB[0,:], dataRGB[3,:])
    return interpX, interpR, interpG, interpB

def funInterpolationContinuousHalftonecurve(vecDigitalInput, dataContinuous, dataHalftoned):
    '''
    The function, obviously, does some interpolation.
    '''
    interpX = np.arange(vecDigitalInput[0], vecDigitalInput[-1]+1)
    #print np.shape(dataContinuous)
    #print np.shape(vecDigitalInput)
    #interpolationContinuous = np.interp(interpX, vecDigitalInput, dataContinuous)
    tck = interpolate.splrep(vecDigitalInput,dataContinuous,s=0)
    interpolationContinuous = interpolate.splev(interpX,tck,der=0)
    #interpolationHalftone   = np.interp(interpX, vecDigitalInput, dataHalftoned)
    tck = interpolate.splrep(vecDigitalInput,dataHalftoned,s=0)
    interpolationHalftone = interpolate.splev(interpX,tck,der=0)

    return interpX, interpolationContinuous, interpolationHalftone

def funInterpolationRatioDifferenceCurves(vecDigitalInput, dataRatioRGB):
    '''
    The function, obviously, does some interpolation, this time for the rati0 experiment.
    '''
    interpX = np.arange(vecDigitalInput[0], vecDigitalInput[-1])
    interpData = np.zeros((3,np.size(interpX)))
    for ii in np.arange(0,3):
        tck = interpolate.splrep(vecDigitalInput,dataRatioRGB[ii,:],s=0)
        interpData[ii,:] = interpolate.splev(interpX,tck,der=0)

    #interpData[0,:] = np.interp(interpX, vecDigitalInput, dataRationRGB[0,:])
    #interpData[1,:] = np.interp(interpX, vecDigitalInput, dataRationRGB[1,:])
    #interpData[2,:] = np.interp(interpX, vecDigitalInput, dataRationRGB[2,:])
    return interpX, interpData

def funInterpolationSingleCurve(data):
    '''
    The function, obviously, does some interpolation.
    data are only one channel (as opposite to the function above)
    but are of the shape data(x,y)
    '''
    #print np.shape(data)
    #print data
    interpX = np.arange(data[0,0], data[0,-1])
    interpY = np.zeros((np.shape(data)[0]-1,np.size(interpX)))
    for ii in np.arange(0,np.shape(data)[0]-1):
        #interpY[ii,:] = np.interp(interpX, data[0,:], data[ii+1,:])
        tck = interpolate.splrep(data[0,:],data[ii+1,:],s=0)
        #print np.shape(tck)
        interpY[ii,:] = interpolate.splev(interpX,tck,der=0)
    return interpX, interpY

def funGetSomeMinimum(interpR, interpG, interpB):
    '''
    Function used by the funGetMinForEachChannel to find the minimum 
    in each interp entree.
    '''
    # get the intensity giving the minimum diff
    minR = np.min(interpR)
    minG = np.min(interpG) 
    minB = np.min(interpB)

    # get the inedx of minimumv
    indR = np.argmin(interpR)
    indG = np.argmin(interpG) 
    indB = np.argmin(interpB)

    return minR, minG, minB, indR, indG, indB

def funGetSomeMinimumSingleCurve(interpM):
    '''
    Function used by the funGetMinForEachChannel to find the minimum 
    in one single interp entree.
    '''
    print 'size interpM is ',np.shape(interpM)
    minM = np.zeros((np.shape(interpM)[0],1))
    indM = np.zeros((np.shape(interpM)[0],1))
    for ii in np.arange(0,np.shape(interpM)[0]):
        # get the intensity giving the minimum diff
        minM[ii] = np.min(interpM[ii,:])
        # get the inedx of minimumv
        indM[ii] = np.argmin(interpM[ii,:])
    return minM, indM

def funDoErrorDiffusion_FloydSteinberg(imageData):
    '''
    As it is written in the name of the function, this function does error error
    diffusion using the Floyd-Steinberg weights to diffuse the errors.
    '''
    imageDataHalftoned = np.zeros(np.shape(imageData))
    #print np.shape(imageData)
    matrice_weight = np.array([[0, 0, 7],[3, 5, 1]]) / 16.
    #print matrice_weight
    for ii in np.arange(1,np.shape(imageDataHalftoned)[0]-1):
        for jj in np.arange(0,np.shape(imageData)[1]-1):
            if imageData[ii,jj] >= 0.5:
                imageDataHalftoned[ii,jj] = 1
                # compute the difference
                pixelDiff = imageData[ii,jj] - 1
            else:
                imageDataHalftoned[ii,jj] = 0
                # compute the difference
                pixelDiff = imageData[ii,jj] - 0
            
            # spread the error
            imageData[ii+1,jj]   = pixelDiff * matrice_weight[0,2] + imageData[ii+1,jj]  # first line
            imageData[ii-1,jj+1] = pixelDiff * matrice_weight[1,0] + imageData[ii-1,jj+1] # second line
            imageData[ii,  jj+1] = pixelDiff * matrice_weight[1,1] + imageData[ii  ,jj+1] # second line
            imageData[ii+1,jj+1] = pixelDiff * matrice_weight[1,2] + imageData[ii+1,jj+1] # second line

    return imageDataHalftoned

def funDoErrorDiffusion_JarvisAndAll(imageData):
    '''
    As it is written in the name of the function, this function does error error
    diffusion using the Jarvis and All weights to diffuse the errors.
    '''
    imageDataHalftoned = np.zeros(np.shape(imageData))
    matrice_weight = np.array([[0., 0., 0., 7., 5.],[3., 5., 7., 5., 3.],[1., 3., 5., 3., 1.]]) / 48.
    for ii in np.arange(2,np.shape(imageDataHalftoned)[0]-2):
        for jj in np.arange(0,np.shape(imageData)[1]-2):
            if imageData[ii,jj] >= 0.5:
                imageDataHalftoned[ii,jj] = 1
                # compute the difference
                pixelDiff = imageData[ii,jj] - 1
            else:
                imageDataHalftoned[ii,jj] = 0
                # compute the difference
                pixelDiff = imageData[ii,jj] - 0
            
            # spread the error
            imageData[ii+1,jj]   = pixelDiff * matrice_weight[0,3] + imageData[ii+1,jj]   # first line
            imageData[ii+2,jj]   = pixelDiff * matrice_weight[0,4] + imageData[ii+2,jj]
            imageData[ii-2,jj+1] = pixelDiff * matrice_weight[1,0] + imageData[ii-2,jj+1]  # second line
            imageData[ii-1,jj+1] = pixelDiff * matrice_weight[1,1] + imageData[ii-1,jj+1]
            imageData[ii,  jj+1] = pixelDiff * matrice_weight[1,2] + imageData[ii  ,jj+1]
            imageData[ii+1,jj+1] = pixelDiff * matrice_weight[1,3] + imageData[ii+1,jj+1]
            imageData[ii+2,jj+1] = pixelDiff * matrice_weight[1,4] + imageData[ii+2,jj+1]

            imageData[ii-2,jj+2] = pixelDiff * matrice_weight[2,0] + imageData[ii-2,jj+2]  # third line
            imageData[ii-1,jj+2] = pixelDiff * matrice_weight[2,1] + imageData[ii-1,jj+2]
            imageData[ii,  jj+2] = pixelDiff * matrice_weight[2,2] + imageData[ii  ,jj+2]
            imageData[ii+1,jj+2] = pixelDiff * matrice_weight[2,3] + imageData[ii+1,jj+2]
            imageData[ii+2,jj+2] = pixelDiff * matrice_weight[2,4] + imageData[ii+2,jj+2]

    ## flip the image to get the best distribution close to the border
    ##imageDataHalftoned = np.fliplr(imageDataHalftoned)
    return imageDataHalftoned

def imCreateTestChartSingleColor(levelRGB):
    '''
    The function takes an RGB value and create an image of the same value.
    '''
    width = 1024
    height = 720
    imagePatchSingleColor = np.zeros((height, width,3), np.uint8)
         
    imagePatchSingleColor[:,:,0] = levelRGB[0]
    imagePatchSingleColor[:,:,1] = levelRGB[1]
    imagePatchSingleColor[:,:,2] = levelRGB[2]
    
    return imagePatchSingleColor

def funDoHalftoningByMaskLinear(im):
    '''
    The function halfotned the image using the mask technique
    The fitler mask are created everytime the function is called.
    '''
    size_mask = 16
    size_image = np.shape(im)

    # check if the image size is a multiple of the mask size
    rest_division_image_size = np.zeros((2,2))
    rest_division_image_size[0,:] = divmod(size_image[0], size_mask)
    rest_division_image_size[1,:] = divmod(size_image[1],size_mask)
    
    # create the mask random
    data_mask = np.arange(0,256)
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 255.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # creat the image mask 
    size_data_mask_image_factor = np.zeros(2)
    if rest_division_image_size[0,1] != 0:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0] + 1
    else:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0]

    if rest_division_image_size[1,1] != 0:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0] + 1
    else:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0]
    image_data_mask = np.tile(data_mask,(size_data_mask_image_factor[0], size_data_mask_image_factor[1]))
    size_data_mask_image = np.shape(image_data_mask)

    # add border to the image before halftoning
    image_with_border = np.zeros((size_data_mask_image[0], size_data_mask_image[1]))
    image_with_border[0:size_image[0], 0:size_image[1]] = im

    # do the halftoning
    image_halftoned = image_with_border <= image_data_mask
    image_halftoned = image_halftoned.astype(float)

    # and we remove the added border:
    image_halftoned = image_halftoned[0:size_image[0],0:size_image[1]]
    
    return image_halftoned

def funDoHalftoningByMaskRandom(im):
    '''
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, only one mask is created.
    '''
    imHalfoned = np.zeros(np.shape(im))
    size_mask = 16
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    #data_mask = np.resize(data_mask,(16,16))
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 255.
    print data_mask, data_mask.dtype
    data_mask = np.resize(data_mask,(size_mask,size_mask))
    print data_mask, data_mask.dtype
    # apply the mask on the image
    vec_X = np.arange(0,np.shape(im)[0],size_mask)
    vec_Y = np.arange(0,np.shape(im)[1],size_mask)
    for ii in vec_X[0:-1]:
        for jj in vec_Y[0:-1]:
            # select the block in the image
            data_block = im[ii:ii+size_mask, jj:jj+size_mask]
            # thresholding
            data_block[data_block <= data_mask] = 0
            data_block[data_block > data_mask] = 1
            imHalfoned[ii:ii+size_mask, jj:jj+size_mask] = data_block
            print data_block
    return imHalfoned

def funDoHalftoningByMaskRandomRGB(imRGB):
    '''
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, only one mask is created.
    '''
    imHalfonedRGB = np.zeros(np.shape(imRGB))
    size_mask = 16
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    data_mask = np.resize(data_mask,(16,16))
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 256.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # apply the mask on the image
    vec_X = np.arange(0,np.shape(imRGB)[0],size_mask)
    vec_Y = np.arange(0,np.shape(imRGB)[1],size_mask)
    for kk in np.arange(0,3):
        for ii in vec_X[0:-1]:
            for jj in vec_Y[0:-1]:
                # select the block in the image
                data_block = imRGB[ii:ii+size_mask, jj:jj+size_mask,kk]
                # thresholding
                data_block[data_block <= data_mask] = 0
                data_block[data_block > data_mask] = 1
                imHalfonedRGB[ii:ii+size_mask, jj:jj+size_mask,kk] = data_block
    return imHalfonedRGB

def funDoHalftoningByMaskFullRandom(im):
    '''
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, for every processed block 
    a new mask is created. But this time we try to be smart.
    '''
    
    size_mask = 16
    size_image = np.shape(im)

    # check if the image size is a multiple of the mask size
    rest_division_image_size = np.zeros((2,2))
    rest_division_image_size[0,:] = divmod(size_image[0], size_mask)
    rest_division_image_size[1,:] = divmod(size_image[1],size_mask)
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 256.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # create the image mask 
    size_data_mask_image_factor = np.zeros(2)
    if rest_division_image_size[0,1] != 0:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0] + 1
    else:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0]

    if rest_division_image_size[1,1] != 0:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0] + 1
    else:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0]
    image_data_mask = np.tile(data_mask,(size_data_mask_image_factor[0], size_data_mask_image_factor[1]))
    size_data_mask_image = np.shape(image_data_mask)

    # add border to the image before halftoning
    image_with_border = np.zeros((size_data_mask_image[0], size_data_mask_image[1]))
    image_with_border[0:size_image[0], 0:size_image[1]] = im

    # do the halftoning
    image_halftoned = image_with_border > image_data_mask
    image_halftoned = image_halftoned.astype(float)

    # and we remove the added border:
    image_halftoned = image_halftoned[0:size_image[0],0:size_image[1]]
    
    return image_halftoned

def funDoHalftoningByMaskFullRandomRGB(imRGB):
    '''
    The same as above but in color
    The function halfotned the image using the mask technic.
    The fitler mask are created everytime the function is called.
    The filter is creating using random number, for every processed block 
    a new mask is created. But this time we try to be smart.
    '''
    
    size_mask = 16
    size_image = np.shape(imRGB)

    # check if the image size is a multiple of the mask size
    rest_division_image_size = np.zeros((2,2))
    rest_division_image_size[0,:] = divmod(size_image[0], size_mask)
    rest_division_image_size[1,:] = divmod(size_image[1], size_mask)
    
    # create the mask random
    data_mask = np.random.rand(size_mask,size_mask)
    data_mask = np.resize(data_mask,(1,size_mask * size_mask))
    data_mask = np.argsort(data_mask)  / 255.
    data_mask = np.resize(data_mask,(size_mask,size_mask))

    # create the image mask 
    size_data_mask_image_factor = np.zeros(2)
    if rest_division_image_size[0,1] != 0:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0] + 1
    else:
        size_data_mask_image_factor[0] = rest_division_image_size[0,0]

    if rest_division_image_size[1,1] != 0:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0] + 1
    else:
        size_data_mask_image_factor[1] = rest_division_image_size[1,0]
    
    # create only one channel then expand
    image_data_mask_one_channel = np.tile(data_mask,(size_data_mask_image_factor[0], size_data_mask_image_factor[1]))
    size_data_mask_image_one_channel = np.shape(image_data_mask_one_channel)
    image_data_mask = np.zeros((size_data_mask_image_one_channel[0], size_data_mask_image_one_channel[1], 3))
    print size_data_mask_image_one_channel
    image_data_mask[:,:,0] = image_data_mask_one_channel
    image_data_mask[:,:,1] = image_data_mask_one_channel
    image_data_mask[:,:,2] = image_data_mask_one_channel

    # add border to the image before halftoning
    image_with_border = np.zeros((size_data_mask_image_one_channel[0], size_data_mask_image_one_channel[1], 3))
    image_with_border[0:size_image[0], 0:size_image[1],:] = imRGB

    # do the halftoning
    image_halftoned = image_with_border <= image_data_mask
    image_halftoned = image_halftoned.astype(float)

    # and we remove the added border:
    image_halftoned = image_halftoned[0:size_image[0],0:size_image[1],:]
    
    return image_halftoned

def funSuperFittFunction(x, A, B, alpha, beta):
    '''
    This function is a super gamma curve swiss knife.
    val = A.x^alpha / (x^beta + B)
    '''
    nume = A*np.power(x/255., alpha) 
    deno = np.power(x/255., beta) + B
    deno = deno + 1e-6 # ugly trick to avoid to divide by 0
    val = nume / deno  
    # and normalization des familles
    val = val / ( (A*np.power(1, alpha)) / (np.power(1, beta) + B))
    return val