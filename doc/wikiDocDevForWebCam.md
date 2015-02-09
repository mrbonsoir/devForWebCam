# What is devForWebCam?

As I mentioned in the README.md document, this project is written in python. The overall goal is to be able to measure the tone response curve (TRC) of a display and some ratio between its color channels. The proper explanation will come later or sooner in a research article. What are described here are the tools to collect data and later analyse them in order to draw some pretty clever - or not - conclusion.

All is based on intensity comparison and/or measurement. A webcam is used as an intensity measuring device. Once can argue that using an uncalibrated webcam is a doom idea to measure the TRC of projector and they are not completely wrong. We take here a projector as the type of display we want to characterize (TRC, color gamut).

We are basing your project on an experiment where normally it is asked to a human observer to evaluate the TRC by the mean of successive patch comparisons. What we propose here is to mimic an eye with a webcam.

# Experimental process

One very important aspect in any experiment is the repeatability. You should be able to reproduce an experiment, to export your experimental processus such that more data can be recorded by different people and further conduct robust statistical analyis.

## Set up your working environment 

Basically you want:

- A dark room, meaning you have control ove the lighting condition, you want ot be sure to measure only the light from the projector
- A projector where ideally once you have performed a series of test on a projector you could switch with another projector model and start a new series of test
- A webcam
- A laptop with some Pyhton 

I usually use a laptop (with ubuntu, these days an X200s) and a projector attached to it as a secondary display without miroring the displays, meaning dual screen mode. With the Python code you are controlling the webcam and the test images. These test images will be displayed on the secondary display (so you may have to move a window to the secondary display).

Your test environment should look a bit like in the illustration below:

![alt text](https://github.com/mrbonsoir/devForWebCam/blob/master/doc/data/standardExperimentalSetup.jpg "standard experimental setup to measure a projector TRC")

## Position your camera and projector

In previous illustration we got an idea of how are positionned the different elements in your living-room, garage or laboratory. 

Everything is in place, you can run the first Python program as follows:

```
python setupFindCameraPosition.py
```

and you can tune the position of you camera Vs projector and projection surface. This code should let appear window with the visible camera streaming and some additionnals information overlapped at each frame. What I'm searching is to have webcam and projector oriented toward the same direction such that the field of view (FOV) of the webcam encopass the projection surface.

![alt text](https://github.com/mrbonsoir/devForWebCam/blob/master/doc/data/printScreenFindCameraPosition.jpg "a print screnn of the control window showing the webcam video stream")

# What to do next?

Now it is time to do measurement! The moment you were waiting for an eternity. Your camera is in place, you just need to use the *computeResponseCurveWithWebcam.py* program. This code allows you to start communication with the webcam and then do measurement. The beginning of the file is for seting up parameters, basically the steps for incrementing the ramp level in orde to sample the RC.

```
python computeResponseCurveWithWebcam.py
```

This Python file is running two different methods to estimate the RC. One is called Method 1 and the second Method 2.

- Method 1 is comparing for every ramp step in its halftoned version a ramp of continuous tone.
- Methos 2 is smarter. Less images are taken and the process is supposed to be faster.