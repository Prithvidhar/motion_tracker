'''
Prithvidhar Pudu 1001570483
CSE-4310-001
Assignment 3

GUI template and main function referenced from: https://github.com/ajdillhoff/CSE4310/blob/main/qtdemo.py
'''
import PIL
from PIL import Image
# import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation
from scipy.spatial import distance
import sys
import random
import argparse
import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_

from PySide2 import QtCore, QtWidgets, QtGui
from skvideo.io import vread


backcount = 0
framecount = 0
# List to keep track of tracked objects
tracked = []
# List to keep track of candidates 
candidates = []
index = 0
nf = 400
# thresh = 0.07
framecount = 0
framecount60 = 0
# The motion detector object's hyperparameters can be tuned here to achieve better motion tracking
class MotionDetector:
    def __init__(self,alpha,thresh,dis, s, N):
        self.alpha = alpha
        self.thresh = thresh
        self.dis = dis
        self.s = s
        self.N = N
md = MotionDetector(alpha = 10, thresh = 0.07, dis = 1.5, s = 1,N = 25)
# Kalman filters that are used to predict and update the position of tracked objects and store previous positions
class KalmanFilter:
    def __init__(self, x, y, vx, vy,alpha,regioncore = 0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.alpha = alpha
        self.history = []
        self.linx = np.array([self.x,self.y,self.vx,self.vy]).reshape(4,1)
        # print(np.array([self.x,self.y,self.vx,self.vy]).reshape(4,1))
        self.covar =  self.linx@self.linx.T
        self.regioncore = regioncore
    def update(self,y):
        deltat = 1
        Dx = np.array( [
            [1,0,deltat,0],
            [0,1,0,deltat],
            [0,0,1,0],
            [0,0,0,1]
        ])
        self.xi = np.eye(Dx.shape[0])
        linxguess = Dx@self.linx
        # covarguess = Dx@(self.covar@Dx.T)
        covarguess = Dx @ self.covar @ Dx.T + self.xi
        ycovar = np.array([[0.1,0],
                        [0,0.1]])
        M = np.array([[1,0,0,0],
                       [0,1,0,0] ])
        
        # print(covarguess+ycovar)
        kGain = covarguess@M.T@(np.linalg.pinv(M @ covarguess @ M.T +ycovar)) # 4x2
        self.linx =linxguess + kGain@(y-M@linxguess) #2x4
        self.covar = covarguess-(kGain@M@covarguess)# 4x4
        self.history.append(np.array([self.x,self.y]).reshape(-1,1))
        # print(self.history)
        self.x = self.linx[0][0]
        self.y = self.linx[1][0]
        self.vx = self.linx[2][0]
        self.vy = self.linx[3][0]

    def predict(self):
        # deltat is change in frames
        deltat = 1
        Dx = np.array( [
            [1,0,deltat,0],
            [0,1,0,deltat],
            [0,0,1,0],
            [0,0,0,1]
        ])
        linxguess = Dx@self.linx
        # print(linxguess)
        covarguess = Dx@self.covar@Dx.T
        # return [linxguess,covarguess]
        return [linxguess[0][0],linxguess[1][0],linxguess[2][0],linxguess[3][0]]

class QtDemo(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()
        self.backout = 0
        self.framecount = 0
        self.framecount60 = 0
        self.frames = frames

        self.current_frame = 0
        # GUI consists of buttons to move md.s frames, move 60 frames in either direction of the video
        # The backward button reset the candidates of the object tracker. Tracking and tracing will continue once the "Next Frame" is clicked repeatedly
        # A label to help users know what frame they are on is also present and will update with every button press.
        self.button = QtWidgets.QPushButton("Next Frame")

        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.button2 = QtWidgets.QPushButton("Previous Frame")
        self.button3 = QtWidgets.QPushButton("Skip 60 Frames")
        self.button4 = QtWidgets.QPushButton("Go Back 60 Frames")
        self.framelabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.framelabel.setText(str(self.current_frame))
        self.framelabel.setFont(QtGui.QFont('Arial',20))
        ##########Inserting my motion detector#########################
        # Calculating initial objects, this motion detector code will be seen often as it is used to find new candidates.
        frame1 = rgb2gray(self.frames[self.current_frame])
        frame2 = rgb2gray(self.frames[self.current_frame+1])
        frame3 = rgb2gray(self.frames[self.current_frame+2])
        framediff1 = abs(frame1-frame2)
        framediff2 = abs(frame2-frame3)
        mot = np.minimum(framediff1,framediff2)
        mot = mot > md.thresh
        dilated_frame = dilation(mot, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        for r in regions:
            candidates.append(KalmanFilter(r.centroid[0],r.centroid[1],1,1,0))
        # Creating motion detector object
        
        #####################################################################

        h, w, c = self.frames[0].shape
        if c == 1:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0]-1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button)
        #Adding "previous frame" button
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)
        self.layout.addWidget(self.button4)
        self.layout.addWidget(self.frame_slider)
        self.layout.addWidget(self.framelabel)

        # Connect functions
        self.button.clicked.connect(self.on_click)
        # Connecting button2 to on_click_back()
        self.button2.clicked.connect(self.on_click_back)
        self.button3.clicked.connect(self.on_click60)
        self.button4.clicked.connect(self.on_click60back)
        self.frame_slider.sliderMoved.connect(self.on_move)


    @QtCore.Slot()
    def on_click(self):
        self.framelabel.setText(str(self.current_frame))
        self.backcount =0
        if self.current_frame+2+md.s >= self.frames.shape[0]-1:
            print('Reached end of available frames scroll back or use the given buttons')
            return
        # print(self.current_frame,'on_click')
        self.framecount += md.s
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        #Inserting code to generate trackers########################
        frame1 = rgb2gray(self.frames[self.current_frame])
        frame2 = rgb2gray(self.frames[self.current_frame+1+md.s])
        frame3 = rgb2gray(self.frames[self.current_frame+2+md.s])
        framediff1 = abs(frame1-frame2)
        framediff2 = abs(frame2-frame3)
        mot = np.minimum(framediff1,framediff2)
        mot = mot > md.thresh
        dilated_frame = dilation(mot, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        newCand = []
        # Comparing 
        for c in candidates:
            for r in regions:
                # If candidate is active update active count else it will become a new candidate.
                # print(np.array(c.predict()))
                cpred = c.predict()
                if distance.euclidean(np.array([cpred[0],cpred[1]]),np.array([r.centroid[0],r.centroid[1]]))<= md.dis:
                    c.alpha =0
                    c.regioncore = np.array([r.centroid[0],r.centroid[1],0,0]).reshape(-1,1)
                else:
                    c.alpha += md.s
                    newCand.append(KalmanFilter(r.centroid[0],r.centroid[1],1,1,0))

        # print('activate candidates check done')
            #Checking for inactive candidates and removing them from list. Adding active candidates to tracked list.
        if self.framecount >= md.alpha:
            
            for c in candidates:
                if c.alpha < md.alpha:
                    if len(tracked) < md.N:
                        tracked.append(c)
                    c.alpha = 0

                else:
                    candidates.remove(c)
        else:
            for c in candidates:
                if c.alpha < md.alpha:
                    if len(tracked) < md.N:
                        tracked.append(c)
                    c.alpha = 0
        # Comparing tracked to candidates
        # print('removed inactive candidates and added new tracked objs')
        # The loop below check for close predections between tracked objects and the candidate
        for t in tracked:
            for c in candidates:
                # print(distance.euclidean(np.array(t.predict()), np.array([c.x,c.y])), md.dis)
                pred = t.predict()
                if distance.euclidean(np.array([pred[0],pred[1]]), np.array([c.x,c.y])) <= md.dis:
                    y = np.array([c.x,c.y]).reshape(-1,1)
                    t.alpha = 0
                    # t.history.append(np.array([t.x,t.y,t.vx,t.vy]).reshape(4,1))
                    # print('update')
                    t.update(y)
                    # if not isinstance(c.regioncore,int):
                    #     t.history.append(c.regioncore)
                    # t.history.append(np.array([c.x,c.y,c.vx,c.vy]).reshape(-1,1))
                    # print('updated tracked objects')
                else:
                    t.alpha += md.s
        # The loop below adds new candidates
        for i in newCand:
            if len(candidates) < md.N:
                candidates.append(i)
        # print('new candidates added')
        # The loop below checks for inactive tracked objects
        # print(self.framecount)
        if self.framecount >= md.alpha:
            # print(framecount,'and', md.alpha)
            for t in tracked:
                if t.alpha > md.alpha:
                    tracked.remove(t)
                    # print('resetting framecount')
                    self.framecount = 0
                else:
                    # t.alpha = 0
                    # print('resetting framecount!!!!')
                    self.framecount = 0
        ##############################################################
        
        
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        # img.scale(0.2, 0.1)
        pixmap = QtGui.QPixmap.fromImage(img)
        qp = QtGui.QPainter(pixmap)
        pen = QtGui.QPen()
        pen.setBrush(QtGui.QColor(255,0,0))
        pen.setWidth(5)
        qp.setPen(pen)
        for mark in tracked:
            if len(mark.history) >1:
                for h in range(len(mark.history)-1):
                    # qp.drawPoint(mark.history[h][1][0],mark.history[h][0][0])
                    qp.drawLine(mark.history[h+1][1][0],mark.history[h+1][0][0],mark.history[h][1][0],mark.history[h][0][0])
            qp.drawPoint(mark.y,mark.x)
        qp.end()
        
        self.img_label.setPixmap(pixmap)
        self.current_frame += 1

    @QtCore.Slot()
    def on_move(self, pos):
        self.framelabel.setText(str(self.current_frame))
        self.current_frame = pos
        print(pos)
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        candidates.clear()
        tracked.clear()
        frame1 = rgb2gray(self.frames[self.current_frame])
        frame2 = rgb2gray(self.frames[self.current_frame+1])
        frame3 = rgb2gray(self.frames[self.current_frame+2])
        framediff1 = abs(frame1-frame2)
        framediff2 = abs(frame2-frame3)
        mot = np.minimum(framediff1,framediff2)
        mot = mot > md.thresh
        dilated_frame = dilation(mot, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        for r in regions:
            candidates.append(KalmanFilter(r.centroid[0],r.centroid[1],1,1,0))
    # Function for moving to previous frame
    @QtCore.Slot()
    def on_click_back(self):
        self.framelabel.setText(str(self.current_frame))
        self.backcount -=1
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:       
            img = QtGui.QImage(self.frames[self.current_frame-1], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame-1], w, h, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(img)
        qp = QtGui.QPainter(pixmap)
        pen = QtGui.QPen()
        pen.setBrush(QtGui.QColor(255,0,0))
        pen.setWidth(5)
        qp.setPen(pen)

        # print(self.backcount)
        for mark in tracked:
            # print(mark.history[0].shape)
            if mark.history and abs(self.backcount) < len(mark.history):
                    qp.drawPoint(mark.history[self.backcount][1][0],mark.history[self.backcount][0][0])
            else:
                print("Re-initialized!")
                    
            
        self.img_label.setPixmap(pixmap)
        qp.end()
        self.current_frame -= 1
        # Re-initializing motion tracker
        candidates.clear()
        tracked.clear()
        frame1 = rgb2gray(self.frames[self.current_frame])
        frame2 = rgb2gray(self.frames[self.current_frame+1])
        frame3 = rgb2gray(self.frames[self.current_frame+2])
        framediff1 = abs(frame1-frame2)
        framediff2 = abs(frame2-frame3)
        mot = np.minimum(framediff1,framediff2)
        mot = mot > md.thresh
        dilated_frame = dilation(mot, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        for r in regions:
            candidates.append(KalmanFilter(r.centroid[0],r.centroid[1],1,1,0))
    @QtCore.Slot()
    def on_click60back(self):
        self.framelabel.setText(str(self.current_frame))
        if self.current_frame <60:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:       
            img = QtGui.QImage(self.frames[self.current_frame-60], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame-60], w, h, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(img)
        self.current_frame -= 60
        self.img_label.setPixmap(pixmap)
        # Re-initializing motion tracker
        candidates.clear()
        tracked.clear()
        frame1 = rgb2gray(self.frames[self.current_frame])
        frame2 = rgb2gray(self.frames[self.current_frame+1])
        frame3 = rgb2gray(self.frames[self.current_frame+2])
        framediff1 = abs(frame1-frame2)
        framediff2 = abs(frame2-frame3)
        mot = np.minimum(framediff1,framediff2)
        mot = mot > md.thresh
        dilated_frame = dilation(mot, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        for r in regions:
            candidates.append(KalmanFilter(r.centroid[0],r.centroid[1],1,1,0))
        
        
    @QtCore.Slot()
    def on_click60(self):
        self.framelabel.setText(str(self.current_frame))
        self.framecount60 += md.s +60
        self.backcount =0
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        #Inserting code to generate trackers########################
        frame1 = rgb2gray(self.frames[self.current_frame+60])
        frame2 = rgb2gray(self.frames[self.current_frame+1+60])
        frame3 = rgb2gray(self.frames[self.current_frame+2+60])
        framediff1 = abs(frame1-frame2)
        framediff2 = abs(frame2-frame3)
        mot = np.minimum(framediff1,framediff2)
        mot = mot > md.thresh
        dilated_frame = dilation(mot, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        newCand = []
        # Comparing 
        for c in candidates:
            for r in regions:
                # If candidate is active update active count else it will become a new candidate.
                # print(np.array(c.predict()))
                cpred = c.predict()
                if distance.euclidean(np.array([cpred[0],cpred[1]]),np.array([r.centroid[0],r.centroid[1]]))<= md.dis:
                    c.alpha += 60+md.s
                else:
                    newCand.append(KalmanFilter(r.centroid[0],r.centroid[1],1,1,0))

        # print('activate candidates check done')
            #Checking for inactive candidates and removing them from list. Adding active candidates to tracked list.
        if self.framecount60 >= md.alpha:
            for c in candidates:
                if c.alpha <= md.alpha:
                    if len(tracked) < md.N:
                        tracked.append(c)
                    c.alpha = 0

                else:
                    candidates.remove(c)
        else:
            for c in candidates:
                if c.alpha <= md.alpha:
                    if len(tracked) < md.N:
                        tracked.append(c)
                    c.alpha = 0
        # Comparing tracked to candidates
        # print('removed inactive candidates and added new tracked objs')
        # The loop below check for close predections between tracked objects and the candidate
        for t in tracked:
            for c in candidates:
                # print(distance.euclidean(np.array(t.predict()), np.array([c.x,c.y])), md.dis)
                pred = t.predict()
                if distance.euclidean(np.array([pred[0],pred[1]]), np.array([c.x,c.y])) <= md.dis:
                    y = np.array([c.x,c.y]).reshape(-1,1)
                    t.alpha += md.s+60
                    # t.history.append(np.array([t.x,t.y,t.vx,t.vy]).reshape(4,1))
                    # print('###########################################')
                    t.update(y)
        # print('updated tracked objects')
        # The loop below adds new candidates
        for i in newCand:
            if len(candidates) < md.N:
                candidates.append(i)
        # print('new candidates added')
        # The loop below checks for inactive tracked objects
        if self.framecount60 >= md.alpha:
            for t in tracked:
                if t.alpha > md.alpha:
                    tracked.remove(t)
                    self.framecount60 = 0
                else:
                    self.framecount60 = 0
        ##############################################################
        
        
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame+60], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame+60], w, h, QtGui.QImage.Format_RGB888)
        # img.scale(0.2, 0.1)
        pixmap = QtGui.QPixmap.fromImage(img)
        qp = QtGui.QPainter(pixmap)
        pen = QtGui.QPen()
        pen.setBrush(QtGui.QColor(255,0,0))
        pen.setWidth(5)
        qp.setPen(pen)
        for mark in tracked:
            qp.drawPoint(mark.y,mark.x)
        qp.end()
        
        self.img_label.setPixmap(pixmap)
        self.current_frame += 60








# Main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    parser.add_argument("--grey", metavar='True/False', type=str, default=False)
    args = parser.parse_args()

    num_frames = args.num_frames

    if num_frames > 0:
        frames = vread(args.video_path, num_frames=num_frames, as_grey=args.grey)
    else:
        frames = vread(args.video_path, as_grey=args.grey)

    app = QtWidgets.QApplication([])

    widget = QtDemo(frames)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())

            
