"""
Modularized detection suite containing the phase_0 classes.
    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""

# Standard library
import copy
import itertools

# External dependencies
import numpy as np
import cv2 as cv

# Project-specific
import utility
from generic import Detector

from config import DEBUG_LEVEL


class BoxExtractor(Detector):
    """This class locates and stores box coordinates in frames."""
    def __init__(self,):
        """"Class constructor. Overrides superclass method."""
        self.set = False

    def detect(self, cur_frame, frame_cnt):
        """Perform box extraction. Overrides superclass method."""
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
            black_lines = np.where(projection <= 80)[0] - OFFSET + 1
            clumps =  np.split(black_lines, np.where(np.diff(black_lines) != 1)[0]+1)
            black_lines = np.where(projection <= 25)[0] - OFFSET + 1
            clumps =  np.split(black_lines, np.where(np.diff(black_lines) != 1)[0] + 1)
            coords = [(clumps[i][-1], clumps[i+1][0]) for i in range(len(clumps) - 1)]
            filtered_coords = [coords[i] for i in np.where(np.diff(coords) > 125)[0]]
            points[axis] = filtered_coords

        if utility.in_range(len(points[0]), 0, 2) and \
           utility.in_range(len(points[1]), 0, 2):
            self.variables['player_regions'][:] = []
            local = list()
            for coord in itertools.product(points[0], points[1]):
                local.append([(int(str(coord[0][0]), 10), int(str(coord[0][1]), 10)),
                              (int(str(coord[1][0]), 10), int(str(coord[1][1]), 10))])
            self.variables['player_regions'] = self.sort_boxes(local, cur_frame)
            if not self.set and self.variables['is_started']:
                self.variables['locked_regions'] = copy.deepcopy(self.variables['player_regions'][0:self.variables['num_players']])
                self.set = True
        else:
            # Completely black frame
            self.variables['player_regions'] = [[(0, cur_frame.shape[1]), (0, cur_frame.shape[0])]]

    def sort_boxes(self, boxes, cur_frame):
        """Sorting algorithm that places priority on "top left" boxes"""
        if len(boxes) != 0:
            ordered = list()
            upper = np.max(boxes).astype(float)
            for box in boxes:
                box_normed = np.divide(box, upper)
                rank = np.sum(100 * box_normed[1]) + np.sum(10 * box_normed[0])
                ordered.append((box, rank))
            ordered = sorted(ordered, key=lambda x:x[1])
            result = [el[0] for el in ordered]
            return result
        else:
            return [[(0, cur_frame.shape[1]), (0, cur_frame.shape[0])]]


class Characters(Detector):
    """Detector for detecting characters in MK64."""
    def __init__(self, masks_dir, freq, threshold, default_shape, buf_len=None):
        """
        Instantiates necessary variables and passes control off to superclass constructor.
        Overrides superclass method.
        """
        self.waiting_black = False
        super(Characters, self).__init__(masks_dir, freq, threshold, default_shape, buf_len)

    def detect(self, frame, cur_count, player):
        """Determines whether and how to process current frame. Overrides superclass method."""
        height, width, _ = frame.shape
        focus_region = frame[np.ceil(height * 0.25) : np.ceil(height * 0.95),
                        np.ceil(width * 0.25) : np.ceil(width * 0.75)]
        if self.variables['is_black']:
            if self.waiting_black:
                self.store_players(focus_region)
            else:
                return
        if not self.variables['is_started'] and (cur_count % self.freq == 0):
            self.process(focus_region, cur_count, player)
            if DEBUG_LEVEL > 2:
                cv.imshow(self.name(), focus_region)
                cv.waitKey(1)

    def process(self, frame, cur_count, player):
        """ Compares pre-loaded masks to current frame. Overrides superclass method."""
        player = 0
        if len(self.default_shape) != 1:
            for mask, shape in zip(self.masks, self.default_shape):
                if frame.shape != shape:
                    scaled_mask = (utility.scaleImage(frame,mask[0], shape), mask[1])
                else:
                    scaled_mask = mask
                distances = cv.matchTemplate(frame, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold:
                    self.handle(frame, player, mask, cur_count, minloc)
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), mask[1], minval)
        else:
            for mask in self.masks:
                if frame.shape != self.default_shape[0]:
                    scaled_mask = (utility.scaleImage(frame,mask[0], self.default_shape[0]), mask[1])
                else:
                    scaled_mask = mask
                distances = cv.matchTemplate(frame, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold:
                    self.handle(frame, player, mask, cur_count, minloc)
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), mask[1], minval)

    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and store variables. Overrides superclass method."""
        self.waiting_black = True
        timestamp = cur_count / self.variables['frame_rate']
        idx = self.buffer.exists(mask[1])
        if idx != -1:
            self.buffer[idx] = (mask[1], location, timestamp)
        else:
            self.buffer.append((mask[1], location, timestamp))

    def store_players(self, frame):
        """Stores players in the state variable."""
        self.waiting_black = False
        self.deactivate()
        ordered = list()
        max_place = 0
        characters = utility.find_unique(self.buffer, 0)

        semi_ordered = self.sort_characters(characters, frame)
        # Find the largest rank in the list.
        for element in semi_ordered:
            if element[1] > max_place:
                max_place = element[1]

        # For every possible player number, find the most recent match
        for ii in xrange(1, (max_place + 1)):
            timestamp = 0
            for element in semi_ordered:
                # Check if the char is player ii and if it was found after <timestamp>
                if element[1] == ii and element[2] > timestamp:
                    timestamp = element[2]
                    temp = (element[0], element[1])
            ordered.append(temp)
        chars = [image[0].split(".", 1)[0] for image in ordered]
        if DEBUG_LEVEL > 0:
            for p, ch in enumerate(chars):
                print "Player %i is %s" % ((p+1), ch)
        self.variables['characters'] = chars
        self.variables['num_players'] = len(ordered)

    def sort_characters(self, characters, frame):
        """Sorting algorithm that ranks based on quadrant."""
        ordered = list()
        center_y, center_x, _ = frame.shape
        center_x /= 2
        center_y /= 2
        for char in characters:
            # Quadrant 2 (Player 1)
            if characters[char][0][1] < (center_x - 10) and characters[char][0][0] < (center_y - 10):
                ordered.append((char, 1, characters[char][1]))
            elif characters[char][0][1] > (center_x + 10) and characters[char][0][0] < (center_y - 10):
                ordered.append((char, 2, characters[char][1]))
            elif characters[char][0][1] < (center_x - 10) and characters[char][0][0] > (center_y + 10):
                ordered.append((char, 3, characters[char][1]))
            else:
                ordered.append((char, 4, characters[char][1]))
        ordered = sorted(ordered, key=lambda x: x[1])
        return ordered


class Map(Detector):
    """Determines which map is being played."""
    def __init__(self, masks_dir, freq, threshold, default_shape, buf_len=None):
        """
        Instantiates necessary variables and passes control off to superclass constructor.
        Overrides superclass method.
        """
        self.waiting_black = False
        self.map = ''
        super(Map, self).__init__(masks_dir, freq, threshold, default_shape, buf_len)

    def detect(self, frame, cur_count, player):
        """Determines whether and how to process current frame. Overrides superclass method."""
        if self.variables['is_black']:
            if self.waiting_black:
                self.variables['map'] = self.map.split('.')[0]
                self.waiting_black = False
                if DEBUG_LEVEL > 0:
                    print 'Locked in map as %s' % (self.map.split('.')[0])
            else:
                return
        if (cur_count % self.freq) == 0:
            self.process(frame, cur_count, player)

    def process(self, frame, cur_count, player):
        """ Compares pre-loaded masks to current frame. Overrides superclass method."""
        best_val = 1
        best_mask = None

        if frame != None:
            # Convert to grayscale and threshold
            binary = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
            _, binary = cv.threshold(binary, 120, 255, cv.THRESH_BINARY)
            binary = cv.GaussianBlur(binary, (3, 3), 1)
            # Scale and grayscale the masks
            if len(self.default_shape) != 1:
                binary_roi = self.constrain_roi(binary)
                for mask, shape in zip(self.masks, self.default_shape):
                    if frame.shape != shape:
                        scaled_mask = (cv.cvtColor(utility.scaleImage(frame, mask[0], shape), 
                            cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary_roi, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                    if DEBUG_LEVEL > 2:
                        cv.imshow('Map Thresh', binary_roi)
                        cv.waitKey(1)
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
            else:
                binary_roi = self.constrain_roi(binary)
                for mask in self.masks:
                    if frame.shape != self.default_shape[0]:
                            scaled_mask = (cv.cvtColor(utility.scaleImage(frame,mask[0], self.default_shape[0]), 
                                cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary_roi, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                    if DEBUG_LEVEL > 2:
                        cv.imshow('Map Thresh', binary_roi)
                        cv.waitKey(1)
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)

    def handle(self, frame, player, mask, cur_count, location):
        """Set variables. Overrides superclass method."""
        self.map = mask[1]
        self.waiting_black = True
        if DEBUG_LEVEL > 0:
            print '[%s]: Map is: %s' % (self.name(), mask[1].split('.')[0])

    def constrain_roi(self, frame):
        """"Constrains frame w.r.t. Maps. Overrides superclass method."""
        h, w = frame.shape
        frame = frame[np.ceil(h * 0.4):, np.ceil(w * 0.45):np.ceil(w * 0.92)]
        return frame


class StartRace(Detector):
    """Handles the beginning of a race in phase_0"""
    def handle(self, frame, player, mask, cur_count, location):
        """Store variables and toggle detectors. Overrides superclass method."""
        self.variables['is_started'] = True
        self.deactivate()
        self.activate('EndRace')
        self.deactivate('Characters')
        self.deactivate('Map')
        # Lock in player boxes (should be sorted alreadY)
        self.variables['player_regions'] = self.variables['player_regions'][0:self.variables['num_players']]
        # Populate dictionary with start time
        self.variables['start_time'] = np.floor(cur_count / self.variables['frame_rate']) - 2
        if DEBUG_LEVEL > 0:
            print '[%s]: Race started at %d seconds' % (self.name(), self.variables['start_time'])
            
    def constrain_roi(self, frame):
        """Constrains frame w.r.t. StartRace/BeginRace. Overrides superclass method."""
        h, w, _ = frame.shape
        frame = frame[0:np.ceil(h * 0.5), np.ceil(w * 0.4):np.ceil(w * 0.75)]
        return frame


class EndRace(Detector):
    """Handles the end of a race (phase_0)"""
    def __init__(self):
        """Class constructor. Overrides superclass method."""
        pass

    def detect(self, frame, cur_count, player):
        """Determines whether and how to process current frame. Overrides superclass method."""
        if self.variables['is_started']:
            if self.variables['is_black']:
                # Either rage-quit or clean race finish (we'll handle rage quits later)
                self.handle(cur_count)
        else:
            self.deactivate()

    def handle(self, cur_count):
        """Store variables and toggle detectors. Overrides superclass method."""
        self.variables['is_started'] = False
        # Populate dictionary with race duration
        self.variables['duration'] = np.ceil((cur_count / self.variables['frame_rate']) - self.variables['start_time'])
        # On end race, deactivate EndRaceDetector, activate StartRaceDetector, BoxExtractor, and CharDetector
        self.deactivate()
        self.activate('StartRace')
        self.activate('Characters')
        self.activate('Map')
        self.activate('BoxExtractor')
        # Store as event for splitting
        self.create_event(start_time=self.variables['start_time'],
                          duration=self.variables['duration'])
        if DEBUG_LEVEL > 0:
            print "[%s] End of race detected at t=%2.2f seconds" % (self.name(), self.variables['duration'])
