""" Revamped detection suite for MK64

    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""

"""Current TODOs:
    * We should probably use Python's logging module in the DEBUG_LEVEL clauses
    * Clean certain sections (just use grep to find #CLEAN labels)
    * Maybe figure out how to reduce time taken by matchTemplate()?
      It's currently the slowest function call, and we use it a lot
    * Split up multiprocessing into its own module (separate from Detection)
    * We only need to reshape masks ONCE per Worker, so we should revert to
      initial implementation for that
    * It might be nice to add a toggled waitKey() into each Worker
      s.t. we can pause those, too
"""


# Standard library
import itertools
import os
import sys

# External dependencies
import numpy as np
import cv2 as cv
import cv2.cv as cv1

# Project-specific
import config
import database
import parallel
import utility

"""DEBUG_LEVEL :Describes intensity of feedback from video processing
    = 0     No feedback
    = 1     Minor feedback      Displaying current frame, print object detections, etc
    = 2     Verbose feedback    Output intermediate values for more severe debugging
    = 3     More verbose        This level will most likely just be used in development
                                of features with unknown results
"""
DEBUG_LEVEL = 2

class Detector(object):
    """Super (and abstract) class for all detectors, written specifically for MK64 events """
    def __init__(self, masks_dir, freq, threshold, default_shape, race_vars, states, buf_len=None):
        if type(self) is Detector:
            raise Exception("<Detector> should be subclassed.")
        self.masks = [(cv.imread(masks_dir+name), name)
                      for name in os.listdir(masks_dir)]
        self.freq = freq
        self.threshold = threshold
        self.default_shape = default_shape
        self.race_vars = race_vars
        self.detector_states = states
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)

    def is_active(self):
        return self.detector_states[self.__class__.__name__]

    def activate(self):
        self.detector_states[self.__class__.__name__] = True

    def deactivate(self):
        self.detector_states[self.__class__.__name__] = False


    def detect(self, frame, cur_count):
        """ Determines whether and how to process current frame """
        if cur_count % self.freq is 0:
            self.process(frame, cur_count)

    def process(self, frame, cur_count):
        """ Compares pre-loaded masks to current frame"""
        for mask in self.masks:
            if frame.shape != self.default_shape:
                scaled_mask = (utility.scaleImage(frame,
                                                  mask[0],
                                                  self.default_shape), mask[1])
            else:
                scaled_mask = (mask[0], mask[1])
            distances = cv.matchTemplate(frame, mask[0], cv.TM_SQDIFF_NORMED)
            minval, _, minloc, _ = cv.minMaxLoc(distances)
            if minval <= self.threshold:
                player = 0 #TODO: Remove this shit
                self.handle(frame, player, mask, cur_count)
                if DEBUG_LEVEL > 1:
                    print "Found %s :-)" % (mask[1])

    def handle(self, frame, player, mask, cur_count):
        # Detectors should be subclassed
        raise NotImplementedError


class BlackFrame(Detector):
    """Faux-detector for determining if frame is black

    Most of the functions are overriding the superclass.
    Updates race variables that race has stopped if above is true
    """
    def __init__(self, race_vars, states):
        self.race_vars = race_vars
        self.detector_states = states

    def detect(self, frame, cur_count):
        self.race_vars.is_black = False
        self.process(frame, cur_count)

    def process(self, frame, cur_count):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
        black_count = float(np.sum(gray)) / float(gray.size)
        # If at least 80% of the frame is true black, race has stopped
        if black_count <= 0.2:
            self.handle(frame, cur_count)

    def handle(self, frame, cur_count):
        self.race_vars.is_black = True
        if DEBUG_LEVEL > 0:
            print "[%s] Handled" % (self.__class__.__name__)


class BoxExtractor(Detector):
    def __init__(self, race_vars, states):
        self.race_vars = race_vars
        self.detector_states = states

    def detect(self, cur_frame, frame_cnt):
        #CLEAN
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

                if DEBUG_LEVEL > 2:
                    cv.imshow('region ' + str(i), cur_frame[col[0]:col[1], row[0]:row[1]])
                    i +=1
            
            # xxx: Must change this code to still switch regions 1 and 2, but do it much cleaner
            if len(self.race_vars.player_boxes) > 3:
                tmp_box = self.race_vars.player_boxes[1]
                self.race_vars.player_boxes[1] = self.race_vars.player_boxes[2]
                self.race_vars.player_boxes[2] = tmp_box

        # DEBUG
        if DEBUG_LEVEL > 2:
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


class Items(Detector):
    """Detector for MK64 items"""
    def handle(self, frame, player, mask, cur_count):
        blank = 'blank_box.png' # Name of image containing blank item box
        self.buffer.append(mask[1])
        last_item = self.buffer[len(self.buffer) - 2]
        # Sorry for the gross if-statemen :-(
        if len(self.buffer) > 1 and mask[1] is blank and last_item is not blank:
            #TODO Update JSON here
            self.buffer.clear()
            if DEBUG_LEVEL > 1:
                print "[%s] Player %s has %s" % (self.__class__.__name__, player, last_item)

class Characters(Detector):
    def __init__(self, masks_dir, freq, threshold, default_shape, race_vars, states, buf_len=None):
        self.waiting_black = False
        super(Characters, self).__init__(masks_dir, freq, threshold, default_shape, race_vars, states, buf_len)

    def detect(self, frame, cur_count):
        # If the race has started, but the detector is still active, deactivate it
        if self.waiting_black and self.race_vars.is_black:
            self.store_players()
        if not self.race_vars.is_started and (cur_count % self.freq is 0):
            player = 0
            for player_box in self.race_vars.player_boxes:
                tmp_frame = frame[player_box[0][0]:player_box[1][0], player_box[0][1]:player_box[1][1]]
                h_frame, w_frame, _ = tmp_frame.shape
                # Focus in on partial frame containing character boxes
                tmp_frame = tmp_frame[np.ceil(h_frame*0.25):np.ceil(h_frame*0.95), np.ceil(w_frame*0.25):np.ceil(w_frame*0.75)]
                self.process(tmp_frame, cur_count)
                player = (player + 1) % 4

                if DEBUG_LEVEL > 1:
                    cv.imshow('char_region', tmp_frame)
                    cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count):
        self.waiting_black = True

    def store_players(self):
        players = utility.find_unique(self.buffer)
        self.waiting_black = False
        print players


class StartRace(Detector):
    def handle(self, frame, player, mask, cur_count):
            self.race_vars.is_started = True
            self.detector_states['StartRace'] = False
            self.detector_states['EndRace']   = True
            self.detector_states['Characters'] = False
            # Populate dictionary with start time
            self.race_vars.race['start_time'] = np.floor(cur_count / self.race_vars.race['frame_rate']) - 6
            if DEBUG_LEVEL > 0:
                print '[StartRace]: Race started at ' + str(self.race_vars.race['start_time']) + ' seconds.'
                cv.waitKey()


class EndRace(Detector):
    def __init__(self, race_vars, states, session_id):
        self.race_vars = race_vars
        self.detector_states = states
        self.session_id = session_id

    def detect(self, frame, cur_count):
        if self.race_vars.is_started:
            if self.race_vars.is_black:
                # Either rage-quit or clean race finish (we'll handle rage quits later)
                self.handle(cur_count)
        else:
            self.detector_states[self.__class__.__name__] = False

    def handle(self, cur_count):
        self.race_vars.is_started = False
        # On end race, deactivate EndRaceDetector, activate StartRaceDetector and CharDetector
        self.detector_states['EndRace']   = False
        self.detector_states['StartRace'] = True
        self.detector_states['Characters'] = True
        if DEBUG_LEVEL > 0:
            print self.detector_states
        # Populate dictionary with race duration
        self.race_vars.race['race_duration'] = np.ceil((cur_count / self.race_vars.race['frame_rate']) - self.race_vars.race['start_time'])
        if DEBUG_LEVEL == 0:
            database.put_race(self.session_id, self.race_vars.race['start_time'], self.race_vars.race['race_duration'])
        else:
            print "[%s] End of race detected at t=%2.2f seconds" % (self.__class__.__name__, self.race_vars.race['race_duration'])
            cv.waitKey()



class Engine():
    """Driving module that feeds Workers video frames"""
    def __init__(self, race_vars, states, video_source):
        self.name = video_source
        self.capture = cv.VideoCapture(video_source)
        self.ret, self.frame = self.capture.read()

        self.race_vars = race_vars
        self.race_vars.race['frame_rate'] = self.capture.get(cv1.CV_CAP_PROP_FPS)
        self.detector_states = states

        self.barrier = None
        self.manager = None

        #DEBUG
        self.toggle = 1
        print '[Engine]: initialization complete.'

    def setup_processes(self, num, regions):
        """Generates child processes"""
        self.barrier = utility.Barrier(parties=(num+1))
        self.manager = parallel.ProcessManager(num, regions, self.frame, self.barrier)

    def add_detectors(self, detect_list):
        """Appends new detectors to Workers, wrapping the ProcessManager"""
        if self.barrier is None:
            raise RuntimeError("You need to call setup_processes() first")
        for detector in detect_list:
            self.detector_states[detector.__class__.__name__] = True
        self.manager.set_detectors(detect_list, self.detector_states)
        self.manager.start_workers()

    def process(self):
        """Loops through video source, feeding child processes new data"""
        frame_count = 0
        while self.frame is not None:
            for i in xrange(parallel.BUFFER_LENGTH):
                offset = i * self.frame.size;
                self.manager.image[offset : offset + self.frame.size] = self.frame.ravel()
                self.ret, self.frame = self.capture.read()
                if self.frame is None:
                    break
                if DEBUG_LEVEL > 0:
                    cv.imshow(self.name, self.frame)
                    frame_count += 1
                    key = cv.waitKey(self.toggle)
                    if key is 27:
                        return
                    elif key is 32:
                        self.toggle ^= 1

            self.manager.detect()
            self.barrier.wait()

    def cleanup(self):
        """Frees memory, alerts child processes to finish"""
        self.barrier.abort()
        self.capture.release()
        self.manager.close()
