# What is devForWebCam?

As I mentioned in the README.md document, this project is written in python. The overall goal is to be able to measure the tone response curve (TRC) of a display and some ratio between its color channels. The proper explanation will come later or sooner in a research article. What are described here are the tools to gather data and later analyse them in order to draw some pretty clever - or not - conclusion.

All is based on intensity comparison and/or measurement. When measurement a webcam is used as an intensity measuring device. Once can argue that using an uncalibrated webcam is a doom idea to measure the TRC of projector - we take here a projector as the type of display we want to characterize - and they are not completely wrong. 

But we are basing your project on an experiment where normally it is asked to a human observer to evaluate the TRC by the mean of successive patch comparisons. What we propose here is to mimic an eye with a webcam.

# Experimental process

One very important aspect in any experiment is the repeatability. You should be able to reproduce an experiment, to export your experimental processus such that more data can be recorded by different people and to conduct robust statistical analyis.

## Set up your working environment 

Basically you want:

- A dark room, meaning you have control ove the lighting condition, you want ot be sure to measure only the light from the projector.
- A projector where ideally once you performed one series of test on a projector you could switch with another projector model.
- A webcam
- A laptop with some pyhton 

I usually use a laptop and a projector attached to it as a secondary display without miroring the displays, meaning dual screen mode. With the code you are controlling the webcam and the test images you will project on the secondary display.

Your test environment should look a bit like in illustration below:

![alt text](https://github.com/mrbonsoir/devForWebCam/blob/master/doc/data/standardExperimentalSetup.jpg "standard experimental setup to measure a projector TRC")

## Position your camera and projector

In previous illustration we got an idea of how are positionned the different elements in your living-room, garage or laboratory. 

Everything is in place, you can run the first python as follows:

```
python setupFindCameraPosition.py
```

to tune the position of you camera vs projector and projection surface. This code should let appear window with the visible camera streaming and some additionnals information overlapped at each frame.

![alt text](https://github.com/mrbonsoir/devForWebCam/blob/master/doc/data/printScreenFindCameraPosition.jpg "a print screnn of the control window showing the webcam video stream")

# What to do next?

Now it is time to do measurement! The moment you were waiting for an eternity. You camera is in place, you just need to use the other python little programs:

- *setupFindResponseCurveAndRatio.py* is a pretty cool tool as it allows you to evaluate the tone response curve of your display. It does more as is also evaluate ratio parameters to do even more.
- *setupFindReponseCurveByHuman.py* does the same as the previous python code, except the final judgement is given by a human feedback.