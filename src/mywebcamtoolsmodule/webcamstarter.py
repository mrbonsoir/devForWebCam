'''A module containing function to access and initialize webcam video stream
'''

import cv2
import numpy as np 
import sys

def fun_start_webcam(webcam_number = 0):
	''' The function create a videocapture object

	kwargs: 
	- webcam_number (int): number to select the camera, by default 0 should give you the embeded camera.

	Output:
	- videocapture object
	'''

	cap = cv2.VideoCapture(webcam_number)

	# display basic info about the stream:
	print 'Stream of size '+str(cap.get(3))+'x'+str(cap.get(4))+'.'

	return cap

def fun_display_webcam_stream(cap, webcam_stream_window_name = "webcam stream"):
	'''The functon display the webcam stream...
	
	kargs:
	- webcam_stream_window_name (char): the name of the window where the stream is displayed

	arg:
	- cap: a videoCapture object
	'''
	# create window to show the stream

	cv2.namedWindow(webcam_stream_window_name, cv2.cv.CV_WINDOW_AUTOSIZE)

	print 'Press q to close the streaming.'

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	   	# Display the resulting frame
	    
	    cv2.imshow(webcam_stream_window_name,frame)
	    if cv2.waitKey(10) == ord('q'):
	        break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

# to run the module as a script
if __name__ == "__main__":
	# create a videocapture element
	if len(sys.argv) == 2:
		# start communication with default camera
		cap = fun_start_webcam(webcam_number =  int(sys.argv[1]))
	else:
		# start commnunication with chosen camera
		cap = fun_start_webcam()

	# and now display the bazard
	fun_display_webcam_stream(cap)
