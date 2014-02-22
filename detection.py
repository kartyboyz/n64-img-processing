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
    def __init__(self, masks_path, freq, threshold, race_vars, default_frame, buf_len=None):
        self.masks = [(cv.imread(masks_path+name), name) for name in os.listdir(masks_path)]
        self.freq = freq
        self.threshold = threshold
        # Set self.scaled to True if masks have already been scaled or if scaling is not necessary.
        self.race_vars = race_vars
        self.frame_shape_default = default_frame
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)
        # Debug
        self.toggle = 0

    def detect(self, frame, cur_count):
        if cur_count % self.freq is 0:
            player = 0
            # Note that this operation should only happen once per detector
            #print self.race_vars.player_boxes
            for player_box in self.race_vars.player_boxes:
                tmp_frame = frame[player_box[0][0]:player_box[1][0], player_box[0][1]:player_box[1][1]]
                self.process(tmp_frame, cur_count, player)
                player = (player + 1) % 4

    def process(self, frame, cur_count, player):
        # Player counter
        for mask in self.masks:
            # If player region (tmp_frame) is not 315x235, scale all masks accordingly
            if frame.shape != self.frame_shape_default:
                scaled_mask = (utility.scaleImage(frame, mask[0], self.frame_shape_default), mask[1])
            else:
                scaled_mask = (mask[0], mask[1])
            # Determine distances
            distance_map = cv.matchTemplate(frame, scaled_mask[0], cv.TM_SQDIFF_NORMED)
            threshold_areas = np.where(distance_map < self.threshold)

            # DEBUG
            for ii in range(len(threshold_areas[0])):
                print 'Mask: ' + scaled_mask[1] + ', Distance: ' + str(distance_map[threshold_areas[0][ii]][threshold_areas[1][ii]])

            if threshold_areas[0].size != 0:
                self.handle(frame, cur_count, player, mask)


class EndRaceDetector(object):
    def __init__(self, session_id, race_vars):
        self.session_id = session_id
        self.race_vars = race_vars

    def detect(self, frame, cur_count):
        # If race hasn't started, still on map selection, or player selection pages, do not process
        if self.race_vars.is_started:
            self.process(frame, cur_count)

    def process(self, frame, cur_count):
         # Threshold for true black
        # XXX/TODO: This is REALLY slow for actual black frames :(
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # ROI is the verticle black line separator. Then threshold.
        _, gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
        # Check frame for all black everything
        black_count = float(np.sum(gray)) / float(gray.size)
        # If at least 80% of the frame is true black, then it's the end of the race
        if black_count <= 0.2:
            self.handle(frame, cur_count)

    def handle(self, frame, cur_count):
        x = 0
        # Set isStarted back to False in order to process another race
        self.race_vars.is_started = False
        # Put the race duration in the dictionary
        self.race_vars.race['race_duration'] = np.ceil((cur_count / self.race_vars.race['frame_rate']) - self.race_vars.race['start_time'])
        #database.put_race(self.session_id, self.race_vars.race['start_time'], self.race_vars.race['race_duration'])
        print '[EndRaceDetector]: End of race detected after ' + str(self.race_vars.race['race_duration']) + ' seconds.'
        cv.waitKey()


class ItemDetector(Detector):
    def handle(self, frame, cur_count, player, mask):
        self.buffer.append(mask[1])
        if self.buffer.all_same():
            print '[ItemDetector]: Player ' + str(player) + ' has ' + mask[1]
            cv.waitKey()
            # Update JSON!
            self.buffer.clear()


class CharacterDetector(Detector):
    def detect(self, frame, cur_count):
        # If the race has started, but the detector is still active, deactivate it
        if not self.race_vars.is_started and (cur_count % self.freq is 0):
            player = 0
            # Note that this operation should only happen once per detector
            #print self.race_vars.player_boxes
            for player_box in self.race_vars.player_boxes:
                tmp_frame = frame[player_box[0][0]:player_box[1][0], player_box[0][1]:player_box[1][1]]
                h_frame, w_frame, _ = tmp_frame.shape
                # Optimize by only taking a specific region (indicated by percentages)
                tmp_frame = tmp_frame[np.ceil(h_frame*0.25):np.ceil(h_frame*0.95), np.ceil(w_frame*0.25):np.ceil(w_frame*0.75)]
                cv.imshow('char_region', tmp_frame)
                self.process(tmp_frame, cur_count, player)
                player = (player + 1) % 4
            #super(CharacterDetector, self).detect(frame, cur_count)

    def handle(self, frame, cur_count, player, mask):
        # If the matched mask isn't already in race_vars.race, add it. Else, do nothing
        if self.race_vars.race[mask[1].split('.')[0].split('_')[1]] != mask[1].split('.')[0].split('_')[0]:
            self.race_vars.race[mask[1].split('.')[0].split('_')[1]] = mask[1].split('.')[0].split('_')[0]
            print 'Player ' + mask[1].split('.')[0].split('_')[1][1] + ' is ' + mask[1].split('.')[0].split('_')[0]
            cv.waitKey()
            # Update JSON!

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
        hor_lines = np.where(hor_projection <= 25)[0]
        ver_lines = np.where(ver_projection <= 25)[0]
        hor_lines = hor_lines-OFFSET+1
        ver_lines = ver_lines-OFFSET+1
        # Partition to extract coordinates
        hor_clumps = np.split(hor_lines, np.where(np.diff(hor_lines) != 1)[0]+1)
        ver_clumps = np.split(ver_lines, np.where(np.diff(ver_lines) != 1)[0]+1)
        # Extract first and last points in sequential range
        hor_coords = [(hor_clumps[i][-1], hor_clumps[i+1][0]) for i in xrange(len(hor_clumps)-1)]
        ver_coords = [(ver_clumps[i][-1], ver_clumps[i+1][0]) for i in xrange(len(ver_clumps)-1)]
        # Filter out noisy data
        hor_coords = [hor_coords[i] for i in np.where(np.diff(hor_coords) > 100)[0]]
        ver_coords = [ver_coords[i] for i in np.where(np.diff(ver_coords) > 100)[0]]
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
                cv.imshow('region ' + str(i), cur_frame[col[0]:col[1], row[0]:row[1]])
                i +=1
            
            # xxx: Must change this code to still switch regions 1 and 2, but do it much cleaner
            if len(self.race_vars.player_boxes) > 3:
                tmp_box = self.race_vars.player_boxes[1]
                self.race_vars.player_boxes[1] = self.race_vars.player_boxes[2]
                self.race_vars.player_boxes[2] = tmp_box

        # DEBUG
        #hh = np.zeros((255, hor_projection.shape[0]))
        #vh = np.zeros((ver_projection.shape[0], 255))
        #for ii in xrange(hor_projection.shape[0]):
        #    if hor_projection[ii] > 1:
        #        if hor_projection[ii]:
        #            hh[0:hor_projection[ii], ii] = 1
        #for ii in xrange(ver_projection.shape[0]):
        #    if ver_projection[ii] > 1:
        #        if ver_projection[ii]:
        #            vh[ii,0:ver_projection[ii]] = 1
        #cv.imshow('v', vh)
        #cv.imshow('h', hh)


class StartRaceDetector(Detector):
    def handle(self, frame, cur_count, player, mask):
            # Set isStarted to True since race has started
            self.race_vars.is_started = True
            # Put the start time of the race into the dictionary
            self.race_vars.race['start_time'] = np.floor(cur_count / self.race_vars.race['frame_rate']) - 6
            print '[StartRaceDetector]: Race started at ' + str(self.race_vars.race['start_time']) + ' seconds.'
            cv.waitKey()

    def detect(self, cur_frame, frame_cnt):
        # If race hasn't started and no characters chosen, initiate detection.
        if not self.race_vars.is_started:
            super(StartRaceDetector, self).detect(cur_frame, frame_cnt)

class RageQuit(Detector):
    def detect(self, cur_frame, frame_cnt):
        if self.race_vars.is_started:
            super(RageQuit, self).detect(cur_frame, frame_cnt)

    def reset(self):
        # Reset state variables so that the next race can be processed
        self.race_vars.is_started = False
        self.race_vars.race['start_time'] = 0
        self.race_vars.race['race_duration'] = 0
        self.race_vars.race['num_players'] = 0
        self.race_vars.race['p1'] = 0
        self.race_vars.race['p2'] = 0
        self.race_vars.race['p3'] = 0
        self.race_vars.race['p4'] = 0

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
        print '[Engine]: initialization complete.'

    def add_detector(self, detector):
        self.detectors.extend(detector)

    def remove_detector(self, index):
        self.detectors.pop(index)

    def process(self):
        while self.cur_frame is not None:
            for ii in range(len(self.detectors)):
                self.detectors[ii].detect(self.cur_frame, self.frame_cnt)
            ret, self.cur_frame = self.capture.read()
            cv.imshow(self.name, self.cur_frame)
            self.frame_cnt += 1
            c = cv.waitKey(self.toggle)
            if c is 27:
                return
            elif c is 32:
                self.toggle ^= 1
