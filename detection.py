""" Revamped detection suite for MK64

    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""

"""Current TODOs:
    * We should probably use Python's logging module in the DEBUG_LEVEL clauses
    * Clean certain sections (just use grep to find #CLEAN labels)
    * Maybe figure out how to reduce time taken by matchTemplate()?
      It's currently the slowest function call, and we use it a lot
    * It might be nice to add a toggled waitKey() into each Worker
      s.t. we can pause those, too
    * We have a couple of almost-God-like classes that have a BUNCH
      of internal variables. We should see if we can decrease this somehow
    * On a similar note, a lot of information is just trickled down to the
      subprocess-level through wrapper functions. It's not necessarily bad,
      but it seems pretty redundant
    * We still don't deal with zoomed-out Lakitu for StartRace!
    * Talk to Michael about new put_race fields
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

from config import DEBUG_LEVEL

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

    def name(self):
        return self.__class__.__name__

    def is_active(self):
        return self.detector_states[self.name()]

    def activate(self):
        self.detector_states[self.name()] = True

    def deactivate(self):
        self.detector_states[self.name()] = False

    def detect(self, frame, cur_count):
        """ Determines whether and how to process current frame"""
        if cur_count % self.freq is 0:
            self.process(frame, cur_count)

    def process(self, frame, cur_count):
        """ Compares pre-loaded masks to current frame"""
        for mask in self.masks:
            if frame.shape != self.default_shape:
                scaled_mask = (utility.resize(frame,
                                                  mask[0],
                                                  self.default_shape), mask[1])
            else:
                scaled_mask = mask
            distances = cv.matchTemplate(frame, mask[0], cv.TM_SQDIFF_NORMED)
            minval, _, minloc, _ = cv.minMaxLoc(distances)
            if minval <= self.threshold:
                player = 0 #TODO: Remove this shit
                self.handle(frame, player, mask, cur_count, minloc)
                if DEBUG_LEVEL > 1:
                    print "Found %s :-)" % (mask[1])

    def handle(self, frame, player, mask, cur_count, location):
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
        if DEBUG_LEVEL > 1:
            print "[%s] Handled" % (self.name())


class BoxExtractor(Detector):
    def __init__(self, race_vars, states):
        self.race_vars = race_vars
        self.detector_states = states

    def detect(self, cur_frame, frame_cnt):
        OFFSET = 10
        # Force black frame to ensure first coord is top left of frame
        border_frame = cv.copyMakeBorder(cur_frame, OFFSET, OFFSET, OFFSET, OFFSET, cv.BORDER_CONSTANT, (0, 0, 0))
        # Treshold + grayscale for binary image
        gray = cv.cvtColor(border_frame, cv.COLOR_BGR2GRAY)
        _, gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
        points = [None] * 2
        # Evaluate black lines & extract box coordinates
        for axis in (0, 1):
            projection = gray.sum(axis=axis)
            projection *= 255/(projection.max() + 1)
            black_lines = np.where(projection <= 25)[0] - OFFSET + 1
            clumps =  np.split(black_lines, np.where(np.diff(black_lines) != 1)[0]+1)
            coords = [(clumps[i][-1], clumps[i+1][0]) for i in range(len(clumps) - 1)]
            filtered_coords = [coords[i] for i in np.where(np.diff(coords) > 125)[0]]
            points[axis] = filtered_coords

        if utility.in_range(len(points[0]), 1, 2) and \
           utility.in_range(len(points[1]), 1, 2):
            #TODO Fix permutations to reflect player number
            self.race_vars.player_boxes[:] = []
            for coord in itertools.product(points[0], points[1]):
                self.race_vars.player_boxes.append([(coord[0][0], coord[0][1]), (coord[1][0], coord[1][1])])
        else:
            # Completely black frame
            self.race_vars.player_boxes.append([(0, 0), (cur_frame.shape[0], cur_frame.shape[1])])


class Items(Detector):
    """Detector for MK64 items"""
    def handle(self, frame, player, mask, cur_count, location):
        blank = 'blank_box.png' # Name of image containing blank item box
        self.buffer.append(mask[1])
        last_item = self.buffer[len(self.buffer) - 2]
        # Sorry for the gross if-statemen :-(
        if len(self.buffer) > 1 and mask[1] is blank and last_item is not blank:
            #TODO Update JSON here
            self.buffer.clear()
            if DEBUG_LEVEL > 1:
                print "[%s] Player %s has %s" % (self.name(), player, last_item)


class Characters(Detector):
    def __init__(self, masks_dir, freq, threshold, default_shape, race_vars, states, buf_len=None):
        self.waiting_black = False
        super(Characters, self).__init__(masks_dir, freq, threshold, default_shape, race_vars, states, buf_len)

    def detect(self, frame, cur_count):
        if self.race_vars.is_black:
            if self.waiting_black:
                self.store_players()
            else:
                return
        if not self.race_vars.is_started and (cur_count % self.freq is 0):
            height, width, _ = frame.shape
            focus_region = frame[np.ceil(height * 0.25) : np.ceil(height * 0.95),
                                 np.ceil(width * 0.25) : np.ceil(width * 0.75)]
            self.process(focus_region, cur_count)
            if DEBUG_LEVEL > 1:
                cv.imshow(self.name(), focus_region)
                cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        self.waiting_black = True
        self.buffer.append((mask[1], location))

    def store_players(self):
        self.waiting_black = False
        self.detector_states[self.name()] = False

        characters = utility.find_unique(self.buffer, 0)
        ordered = self.sort_characters(characters)
        for player, image in enumerate(ordered):
            char = image[0].rsplit(".", 1)[0]
            self.race_vars.race["p" + str(player + 1)] = char
            if DEBUG_LEVEL > 0:
                print "Player %d is %s!" % (player + 1, char)
        self.race_vars.race_vars['num_players'] = len(ordered)

    def sort_characters(self, characters):
        """Sorting algorithm which places priority on "top left" players"""
        ordered = list()
        upper = np.max(characters.values()).astype(float)
        for char in characters:
            loc_normed = characters[char] / upper
            rank = 10 * loc_normed[0] + 100 * loc_normed[1]
            ordered.append((char, rank))
        ordered = sorted(ordered, key=lambda x: x[1])
        return ordered


class StartRace(Detector):
    def handle(self, frame, player, mask, cur_count, location):
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
            self.detector_states[self.name()] = False

    def handle(self, cur_count):
        self.race_vars.is_started = False
        # On end race, deactivate EndRaceDetector, activate StartRaceDetector and CharDetector
        self.detector_states['EndRace']   = False
        self.detector_states['StartRace'] = True
        self.detector_states['Characters'] = True
        if DEBUG_LEVEL > 1:
            print self.detector_states
        # Populate dictionary with race duration
        self.race_vars.race['duration'] = np.ceil((cur_count / self.race_vars.race['frame_rate']) - self.race_vars.race['start_time'])
        if DEBUG_LEVEL == 0:
            try:
                database.put_race(self.race_vars)
            except: #TODO Figure out exact exceptions
                # Database error, dump to filesystem
                self.race_vars.save("dbfail_session%i.dump" % (self.race_vars.race['session_id']))
                #TODO Decide if we ever want to be able to recover dump files
        else:
            print "[%s] End of race detected at t=%2.2f seconds" % (self.name(), self.race_vars.race['race_duration'])
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
        print "[%s] initialization complete." % (self.__class__.__name__)

    def setup_processes(self, num, regions):
        """Generates child processes"""
        self.barrier = utility.Barrier(parties=(num+1))
        self.manager = parallel.ProcessManager(num, regions, self.frame, self.barrier, self.race_vars)

    def add_detectors(self, detect_list):
        """Appends new detectors to Workers, wrapping the ProcessManager"""
        if self.barrier is None:
            raise RuntimeError("You need to call setup_processes() first")
        for detector in detect_list:
            self.detector_states[detector.name()] = True
        self.manager.set_detectors(detect_list, self.detector_states)
        self.manager.start_workers()

    def process(self):
        """Loops through video source, feeding child processes new data"""
        frame_count = 0
        size = self.frame.size
        while True:
            try:
                for i in range(parallel.BUFFER_LENGTH):
                    offset = i * size;
                    self.manager.image[offset : offset + size] = self.frame.ravel()
                    self.ret, self.frame = self.capture.read()
                    if not self.ret:
                        self.clear_buffer(offset=offset + size + 1)
                        raise StopIteration
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
            except StopIteration:
                # Handle dangling frames in buffer and return gracefully
                self.manager.detect()
                return

    def clear_buffer(self, offset):
        """Cleans up the rest of the buffer

        Needed so that Workers don't process same data many times
        """
        self.manager.image[offset:] = 0

    def cleanup(self):
        """Frees memory, alerts child processes to finish"""
        self.barrier.abort()
        self.capture.release()
        self.manager.close()
