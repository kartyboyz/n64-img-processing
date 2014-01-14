#!/usr/bin/env python
import collections
import database
from math import floor, ceil
import os
import sys

import numpy as np
import cv as cv1
import cv2 as cv

import utility

# Global flags
isStarted = False

# Constants
BLACK_FRAME_THRESHOLD = 6500000
BLACK_PXL_THRESHOLD = (50,50,50)
TRUE_BLACK = (0,0,0)

# JSON Container for race information
race =    {'num_players':   0,
           'session_id':    0,
           'start_time':    0,
           'race_duration': 0,
           'frame_rate':    0
          }

# Parent class for all detectors
class Detector(object):
    def __init__(self, ROI_list, masks_path, freq, threshold, buf_len=None):
        self.ROI_list = ROI_list
        self.masks = [(cv.imread(masks_path+name), name) for name in os.listdir(masks_path)]
        self.freq = freq
        self.threshold = threshold
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)
        # Debug
        self.toggle = 0

    def detect(self, frame, cur_count):
        if cur_count % self.freq is 0:
            self.process(frame, cur_count)

    def process(self, frame, cur_count):
        # Player counter
        player = 1
        for ROI in self.ROI_list:
            # Extract ROI points
            col0 = ROI[0][0]
            col1 = ROI[1][0]
            row0 = ROI[0][1]
            row1 = ROI[1][1]
            # Pixelate current ROI in frame
            region = frame[row0:row1, col0:col1]
            f_pxl, f_disp = utility.pixelate(region, resolution=8)
            for mask in self.masks:
                # Ignore black pixels in mask
                tmp_frame = f_pxl.copy()
                tmp_frame[(mask[0] <= BLACK_PXL_THRESHOLD).all(axis = -1)] = TRUE_BLACK
                mask[0][(mask[0] <= BLACK_PXL_THRESHOLD).all(axis = -1)] = TRUE_BLACK
                # Debug
                cv.imshow('FRAME', f_disp)
                # Determine distances
                # Determine distances
                distance = cv.matchTemplate(tmp_frame, mask[0], cv.TM_SQDIFF_NORMED)
                print mask[1], distance
                if distance < self.threshold:
                    self.handle(frame, cur_count, player, mask)
                # DEBUG
                cv.imwrite('cur_f.png', tmp_frame)
                cv.imwrite('cur_m.png', mask[0])
            player += 1

class EndRaceDetector(object):
    def __init__(self, session_id):
        self.session_id = session_id

    def detect(self, frame, cur_count):
        # If race hasn't started, still on map selection, or player selection pages, do not process
        if isStarted:
            self.process(frame, cur_count)

    def process(self, frame, cur_count):
         # Threshold for true black
        # XXX/TODO: This is REALLY slow for actual black frames :(
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # ROI is the verticle black line separator. Then threshold.
        gray = gray[239, :]
        gray[gray <= 50] = 0
        h = gray.shape[0]
        # Check ROI for black lines
        black_count_h = np.sum(gray == 0)
        # Using w-40 as threshold to lower false positive rate
        if black_count_h <= h - 40:
            self.handle(frame, cur_count)

    def handle(self, frame, cur_count):
        x = 0
        global isStarted
        global race
        # Set isStarted back to False in order to process another race
        isStarted = False
        # Put the race duration in the dictionary
        race['race_duration'] = ceil((cur_count / race['frame_rate']) - race['start_time']) + 7
        print race['race_duration']
        database.put_race(self.session_id, race['start_time'], race['race_duration'])
        print 'End of race detected'
        cv.waitKey()


class ItemDetector(Detector):
    def handle(self, frame, cur_count, player, mask):
        self.buffer.append(mask[1])
        if self.buffer.all_same():
            print '\t\t\tPlayer ' + str(player) + ' has ' + mask[1]
            cv.waitKey()
            # Update JSON!
            self.buffer.clear()

class PlayerNumDetector(object):
    def __init__(self):
        # Flags
        self.done = False

    def detect(self, frame, cur_count):
        if not self.done:
            # Normalize frame
            frame *= 255.0/frame.max()
            if np.sum(frame) > BLACK_FRAME_THRESHOLD:
                # Not black screen, check for lines
                self.process(frame, cur_count)

    def process(self, frame, cur_count):
        ''' Modes:
                1 player:       No black lines intersecting video stream
                2 players:      One long horizontal black line in video stream
                3, 4 players:   Horizontal and vertical black lines in video stream
        '''
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        (h, w) = gray.shape
        # MK64-specific constants
        LINE_PX_H = h/2
        LINE_PX_W = w/2-3 # NOTE: -3 necessary because of asymmetry in frame
        # Threhsold for true black
        # Check ROI for black lines
        v_test = h_test = False
        black_count_w = np.sum(gray[:, LINE_PX_W] == 0)
        black_count_h = np.sum(gray[LINE_PX_H, :] == 0)
        if black_count_w >= h/2:
            h_test = True
        if black_count_h >= w/2:
            v_test = True
        if not v_test and h_test:
            print '2P mode'
        elif all((h_test, v_test)):
            print '3P or 4P mode'
        else:
            print '1p mode'

class StartRaceDetector(Detector):
    def handle(self, frame, cur_count, player, mask):
            x=0
            global isStarted
            global race
            # Set isStarted to True since race has started
            isStarted = True
            # Put the start time of the race into the dictionary
            race['start_time'] = floor(cur_count / race['frame_rate']) - 6
            print race['start_time']
            print '\t\tRace has started'
            cv.waitKey()

    def detect(self, cur_frame, frame_cnt):
        # If race hasn't started, initiate detection.
        if not isStarted:
            super(StartRaceDetector, self).detect(cur_frame, frame_cnt)

class RageQuit(Detector):
    def detect(self, cur_frame, frame_cnt):
        if isStarted:
            super(RageQuit, self).detect(cur_frame, frame_cnt)

    def reset(self):
        global isStarted
        global race
        # Reset state variables so that the next race can be processed
        isStarted = False
        race['start_time'] = 0
        race['race_duration'] = 0
        race['num_players'] = 0

    def handle(self, frame, cur_count, player, mask):
        print 'Rage Quit Detected'
        self.reset()
        cv.waitKey()

class Engine(object):
    def __init__(self, src):
        global race
        self.name = src
        self.capture = cv.VideoCapture(src)
        race['frame_rate'] =  self.capture.get(cv1.CV_CAP_PROP_FPS)
        self.ret, self.cur_frame = self.capture.read()
        self.avg_frames = np.float32(self.cur_frame)
        self.frame_cnt = 1
        self.detectors = []
        # Debug
        cv.namedWindow(src, 1)
        self.toggle = 1
        print '[Engine] initialization complete.'

    def add_detector(self, detector):
        self.detectors.extend(detector)

    def process(self):
        # Init
        while self.cur_frame is not None:
            print '----[ Frame ' + str(self.frame_cnt) + ']----'
            cv.imshow(self.name, self.cur_frame)
            for d in self.detectors:
                d.detect(self.cur_frame, self.frame_cnt)
            ret, self.cur_frame = self.capture.read()
            self.frame_cnt += 1
            c = cv.waitKey(self.toggle)
            if c is 27:
                return
            elif c is 32:
                self.toggle ^= 1
