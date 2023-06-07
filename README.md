# Motion Tracking Application
## Description
This application allows users to track movement in their video files by displaying trail marks behind moving objects.
## Installation
- The following python packages are needed to run this application:
  1. Numpy
  2. Matplotlib
  3. Skvideo
  4. Skimage
  5. Scipy
  6. QtCore, QtWidgets and QtGui
- It is recommened to use the Anaconda package manager to install these dependencies.
- After installing the packages, clone this repository to your local system and use the following command to run the program: `python motion_tracking.py [-h] [--num_frames n] [--grey True/False] PATH_TO_VIDEO`
## Usage instructions
The GUI consists of a panel that displays your video frames and below it is a variety of UI elements to help you traverse these frames. Repeatedly pressing the "Next Frame" button will load the next frame into the panel. During this process a trail will be generated showing the paths of any moving objects in the video. The other buttons allow the user to skip more frames and track their object at any required frame number.

