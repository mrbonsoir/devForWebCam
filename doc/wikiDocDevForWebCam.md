# What is devForWebCam?

As I mentioned in the README.md document, this project is written in python. The overall goal is to be able to measure the tone response of a display and some ration between its color channels. All is based on intensity comparison and/or measurement. When measurement a webcam is used as an intensity measuring device. 

Once can argue that using an uncalibrated webcam is a doom idea to measure the TRC of projector (we take here a projector as the type of display we want to characterize)

It is a project badly written in python and using opencv to access wihout too many hassle a webcam. Not only opencv, numpy, matplotlib and some more color tools are used as well.

# What whill you find in this project?

There is an attempt of explanation of what you can do with the following python code located in the src folder:

* *webcamTools.py* contains the functions or tools I used in this project.
* *setupFindCameraPosition.py* is the first thing you should try. It allows you to test the communication with the camera. 
* *setupFindResponseCurveAndRatio.py* is a pretty cool tool as it allows you to evaluate the tone response curve of your display. It does more as is also evaluate ratio parameters to do even more.
* *setupFindReponseCurveByHuman.py* does the same as the previous python code, except the final judgement is given by a human feedback (are two patches side by side equivalent in intensity or not).

# What to do next?

You can now roll up your sleeves, look for a usb webcam, a projector and starting measuring and messing around.

There are additionnals information in the doc folder. So you know where to look.


