#!/usr/bin/env python
import collections
import database
import itertools
import os
import sys

import numpy as np
import cv2 as cv
import cv2.cv as cv1

import utility
import config

# Parent class for all detectors
class Detector(object):
    def __init__(self, masks_path, freq, threshold, race_vars, buf_len=None):
        self.masks = [(cv.imread(masks_path+name), name) for name in os.listdir(masks_path)]
        self.freq = freq
        self.threshold = threshold
        # Set self.scaled to True if masks have already been scaled or if scaling is not necessary.
        self.scaled = False
        self.race_vars = race_vars
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)
        # Debug
        self.toggle = 0

    def detect(self, frame, cur_count):
        if cur_count % self.freq is 0:
            # Note that this operation should only happen once per detector
            ii = 0
            for player_box in self.race_vars.player_boxes:
                tmp_frame = frame[player_box[0][0]:player_box[1][0], player_box[0][1]:player_box[1][1]]
                cv.imshow('player' + str(ii), tmp_frame)
                ii ^= 1
                # If mask has not been scaled yet, do it.
                if self.scaled == False:
                    # If player region (tmp_frame) is not 315x235, scale all masks accordingly
                    if tmp_frame.shape != (315, 235, 3):
                        for ii in range(len(self.masks)):
                            scaled_mask = utility.scaleImage(tmp_frame, self.masks[ii][0])
                            self.masks[ii] = (scaled_mask, 'scaled_'+self.masks[ii][1])
                    self.scaled = True
                self.process(tmp_frame, cur_count)

    def process(self, frame, cur_count):
        # Player counter
        player = 1
        for mask in self.masks:
            # Determine distances
            distance_map = cv.matchTemplate(frame, mask[0], cv.TM_SQDIFF_NORMED)
            threshold_areas = np.where(distance_map < self.threshold)
            #print threshold_areas
            for ii in range(len(threshold_areas[0])):
                print mask[1], distance_map[threshold_areas[0][ii]][threshold_areas[1][ii]]
            if threshold_areas[0].size != 0:
                self.handle(frame, cur_count, player, mask)
        player += 1


class EndRaceDetector(object):
    def __init__(self, race_vars, session_id):
        self.session_id = session_id
        self.race_vars = race_vars

    def detect(self, frame, cur_count):
        # If race hasn't started, still on map selection, or player selection pages, do not process
        if self.race_vars.isStarted:
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
        # Set isStarted back to False in order to process another race
        self.race_vars.isStarted = False
        # Put the race duration in the dictionary
        self.race_vars.race['race_duration'] = np.ceil((cur_count / self.race_vars.race['frame_rate']) - self.race_vars.race['start_time']) + 7
        print self.race_vars.race['race_duration']
        database.put_race(self.session_id, self.race_vars.race['start_time'], self.race_vars_race['race_duration'])
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

class BoxExtractor(object):
    def __init__(self, race_vars):
        self.race_vars = race_vars

    def detect(self, cur_frame, frame_cnt):
        OFFSET = 10
        # Force black frame to ensure first coord is top left
        border_frame = cv.copyMakeBorder(cur_frame, OFFSET, OFFSET, OFFSET, OFFSET, cv.BORDER_CONSTANT, (0, 0, 0))
        # Treshold + grayscale for binary image
        gray = cv.cvtColor(border_frame, cv.COLOR_BGR2GRAY)
        _, gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
        # Sum up rows/columns for 'black' projections
        hor_projection = gray.sum(axis=0)
        ver_projection = gray.sum(axis=1)
        # Normalize to [1-255]
        hor_projection *= 255/(hor_projection.max()+1)
        ver_projection *= 255/(ver_projection.max()+1)
        # Extract black line projection indices
        hor_lines = np.where(hor_projection == 0)[0]
        ver_lines = np.where(ver_projection == 0)[0]
        hor_lines = hor_lines-OFFSET+1
        ver_lines = ver_lines-OFFSET+1
        # Partition to extract coordinates
        hor_clumps = np.split(hor_lines, np.where(np.diff(hor_lines) != 1)[0]+1)
        ver_clumps = np.split(ver_lines, np.where(np.diff(ver_lines) != 1)[0]+1)
        # Extract first and last points in sequential range
        hor_coords = [(hor_clumps[i][-1], hor_clumps[i+1][0]) for i in xrange(len(hor_clumps)-1)]
        ver_coords = [(ver_clumps[i][-1], ver_clumps[i+1][0]) for i in xrange(len(ver_clumps)-1)]
        # Filter out noisy data
        hor_coords = [hor_coords[i] for i in np.where(np.diff(hor_coords) > 50)[0]]
        ver_coords = [ver_coords[i] for i in np.where(np.diff(ver_coords) > 50)[0]]
        hor_len = len(hor_coords)
        ver_len = len(ver_coords)

        i = 0 # DEBUG
        self.race_vars.player_boxes[:] = []
        if (hor_len <= 2 and hor_len > 0 and
            ver_len <= 2 and ver_len > 0):
            # Create all permutations for player regions
            ranges = []
            for perm in itertools.product(hor_coords, ver_coords):
                ranges.append((perm[0], perm[1]))
            for (row, col) in ranges:
                # Update global configuration settings
                self.race_vars.player_boxes.append([(col[0], row[0]), (col[1], row[1])])
                # DEBUG
                cv.imshow(str(i), cur_frame[col[0]:col[1], row[0]:row[1]])
                i +=1

        # DEBUG
        hh = np.zeros((255, hor_projection.shape[0]))
        vh = np.zeros((ver_projection.shape[0], 255))
        for ii in xrange(hor_projection.shape[0]):
            if hor_projection[ii] > 1:
                if hor_projection[ii]:
                    hh[0:hor_projection[ii], ii] = 1
        for ii in xrange(ver_projection.shape[0]):
            if ver_projection[ii] > 1:
                if ver_projection[ii]:
                    vh[ii,0:ver_projection[ii]] = 1
        cv.imshow('v', vh)
        cv.imshow('h', hh)


class StartRaceDetector(Detector):
    def __init__(self, masks_path, freq, threshold, race_vars, buf_len=None):
        self.numStarts = 0
        super(StartRaceDetector, self).__init__(masks_path, freq, threshold, race_vars, buf_len=None)

    def handle(self, frame, cur_count, player, mask):
            # Set isStarted to True since race has started
            self.race_vars.isStarted = True
            self.numStarts += 1
            # Put the start time of the race into the dictionary
            self.race_vars.race['start_time'] = np.floor(cur_count / self.race_vars.race['frame_rate']) - 6
            print str(self.race_vars.race['start_time']) + '\t# of players: ' + str(self.numStarts)
            print '\t\tRace has started'
            cv.waitKey()

    def detect(self, cur_frame, frame_cnt):
        # If race hasn't started, initiate detection.
        if not self.race_vars.isStarted:
            super(StartRaceDetector, self).detect(cur_frame, frame_cnt)

class RageQuit(Detector):
    def detect(self, cur_frame, frame_cnt):
        if self.race_vars.isStarted:
            super(RageQuit, self).detect(cur_frame, frame_cnt)

    def reset(self):
        # Reset state variables so that the next race can be processed
        self.race_vars.isStarted = False
        self.race_vars.race['start_time'] = 0
        self.race_vars.race['race_duration'] = 0
        self.race_vars.race['num_players'] = 0

    def handle(self, frame, cur_count, player, mask):
        print 'Rage Quit Detected'
        self.reset()
        cv.waitKey()

class Engine(object):
    def __init__(self, src, race_vars):
        self.name = src
        self.capture = cv.VideoCapture(src)
        self.race_vars = race_vars
        self.race_vars.race['frame_rate'] =  self.capture.get(cv1.CV_CAP_PROP_FPS)
        self.ret, self.cur_frame = self.capture.read()

        gray = cv.cvtColor(self.cur_frame, cv.COLOR_BGR2GRAY)

        self.prev_frame = gray.copy()
        self.frame_cnt = 1
        self.detectors = []
        # Debug
        cv.namedWindow(src, 1)
        self.toggle = 1
        print '[Engine] initialization complete.'

    def add_detector(self, detector):
        self.detectors.extend(detector)

    def process(self):
        while self.cur_frame is not None:
            for d in self.detectors:
                d.detect(self.cur_frame, self.frame_cnt)
            ret, self.cur_frame = self.capture.read()
            cv.imshow(self.name, self.cur_frame)
            self.frame_cnt += 1
            c = cv.waitKey(self.toggle)
            if c is 27:
                return
            elif c is 32:
                self.toggle ^= 1
