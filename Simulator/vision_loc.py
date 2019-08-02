import numpy as np
import scipy.special as sp
import os
import ctypes
import pygame
from math import *
import random as rnd
from screen import *
import time

class VISION():
    #----------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, robot, clk=30.):
        self.robot = robot # Saves the robot object's address
        # print self.robot.x, self.robot.y
        self.robotheight = 50 # Height in centimeters

        self.bkb = self.robot.bkb # Holds Blackboard's address
        self.Mem = self.robot.Mem # Holds memory key

        self.clk = clk

        self.headtilt = 17.4 # Tilt position
        self.tiltpos = 17.4 # Chosen tilt position, default 17.4
        self.headpan = -90 # Pan position
        self.panpos = 0 # Chosen pan position

        self.maxtilt = 64 # Max tilt
        self.mintilt = 16 # Min tilt
        self.maxpan = 90 # Max pan
        self.minpan = -90 # Min pan

        self.headspd = 90 # Max head speed, in degrees per second

        # Observation points
        self.behave = 0

        self.text = ""
        self.index = 0
        self.count = -1

        self.timestamp = 0

    # Changes the tilt's position
    def tilt(self, diff=None, pos=None):
        if diff: # Adds a difference
            self.tiltpos += diff
        elif pos: # Changes the position
            self.tiltpos = pos

    # Changes the pan's position
    def pan(self, diff=None, pos=None):
        if diff != None: # Adds a difference
            self.panpos += diff
        elif pos != None: # Changes the position
            self.panpos = pos

    # Updates the head position
    def headmotion(self):
        self.headtilt += np.sign(self.tiltpos-self.headtilt) * np.min([abs(self.tiltpos-self.headtilt), self.headspd]) / self.clk
        self.headpan += np.sign(self.panpos-self.headpan) * np.min([abs(self.panpos-self.headpan), self.headspd]) / self.clk

        self.headtilt = np.min([self.maxtilt, np.max([self.mintilt, self.headtilt])])
        self.headpan = np.min([self.maxpan, np.max([self.minpan, self.headpan])])

    # Reactive head behavior
    def headBehave(self):
        hp = self.bkb.read_int(self.Mem, 'DECISION_LOCALIZATION')

        if hp == -999:
            # y = [90,60,30,0,-30,-60,-90]
            # y = [90, 0, -90, 0]
            y = [0]
            x = y[self.behave]
            self.pan(pos=x)

            if np.abs(self.headpan-x) < 1:
                self.behave = (self.behave + 1) % len(y)
                if time.time() - self.timestamp >= 1:
                    self.get = True
                    self.timestamp = time.time()
        else:
            if hp != 999:
                self.pan(pos=hp)

            if np.abs(self.headpan-hp) < 1:
                self.get = True
                self.bkb.write_int(self.Mem, 'DECISION_LOCALIZATION', 999)

    # Draw the field of view of the robot
    def draw(self, where):
        # # Computes distances
        # f = self.robotheight/np.tan(np.radians(self.headtilt-self.vfov))
        # n = self.robotheight/np.tan(np.radians(self.headtilt+self.vfov))

        # # Computes point distances
        # dn = n/np.cos(np.radians(self.hfov))
        # df = f/np.cos(np.radians(self.hfov))

        # # Array of points
        # points = []

        # # Create points
        # for d in [dn, df]:
        #     for a in [self.hfov, -self.hfov]:
        #         x = self.robot.x + d * np.cos(np.radians(-self.robot.rotate + self.headpan + a))
        #         y = self.robot.y + d * np.sin(np.radians(-self.robot.rotate + self.headpan + a))
        #         points.append((x,y))

        # # Draw the points
        # for a in [0, 3]:    
        #     for b in [1, 2]:
        #         pygame.draw.line(where, (255,255,255), points[a], points[b], 1)

        # for point in v:
        #     # Compute the point 

        #     dist = point[0] #+ np.random.normal(0,point[0][0]/10.)
        #     # Compute the direction of the point
        #     angle = -self.robot.rotate + self.headpan + point[1] #+ np.random.normal(0,2)

        #     # Compute the position in the world of the point
        #     x = self.robot.x + dist * np.cos(np.radians(angle)) #+ np.random.normal(0,1)
        #     y = self.robot.y + dist * np.sin(np.radians(angle)) #+ np.random.normal(0,1)

        #     pygame.draw.circle(where, self.robot.color, (int(x),int(y)), 2, 0)
            # pygame.draw.circle(where, (0,0,0), (int(x),int(y)), 2, 0)

        pass

    # Return the distance and direction of a given point
    def GetPoint(self, point):
        # Compute the vectorial distance
        dx = point[0] - self.robot.x
        dy = self.robot.y - point[1]

        # Computes the scalar distance
        dist = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx))-self.robot.rotate

        # Normalizes angle
        if ang < -180:
            ang += 360
        elif ang > 180:
            ang -= 360

        # Verifies if the angle is inside the field of view
        if -self.hfov-self.headpan <= ang and ang <= self.hfov-self.headpan:
            # 
            f = self.robotheight/np.tan(np.radians(self.headtilt-self.vfov))
            n = self.robotheight/np.tan(np.radians(self.headtilt+self.vfov))

            # Computes point distances
            dn = n/np.cos(np.radians(ang+self.headpan))
            df = f/np.cos(np.radians(ang+self.headpan))

            # Verifies if the distance is in range
            if (1 + sp.erf((min(df,700)-dist)/(np.sqrt(2)*10))) * (1 - sp.erf((max(10,dn)-dist)/(np.sqrt(2)*1))) / 4 >= np.random.random():
                return np.random.normal(ang, 1)
        return -999

    # Return the position of the ball, maybe
    def GetBall(self):
        dist = hypot(self.robot.x-self.robot.ball.x, self.robot.y-self.robot.ball.y)
        ang = -degrees(atan2(self.robot.ball.y-self.robot.y, self.robot.ball.x-self.robot.x))-self.robot.rotate
        if CompAng(ang, 0, self.fov/2.0):
            return dist, ang
        else:
            return -1, -1

    # Vision Process!
    def VisionProcess(self):
        self.headBehave()
        self.headmotion()


def CompAng(ang, base, rng):
    xa = cos(radians(ang))
    ya = sin(radians(ang))
    xb = cos(radians(base))
    yb = sin(radians(base))
    xr = cos(radians(rng))
    yr = sin(radians(rng))

    d = hypot(xa-xb, ya-yb)
    c = hypot(xr-1, yr)

    return d < c
