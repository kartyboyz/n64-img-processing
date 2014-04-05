''' Modularized detection suite containing the phase_0 classes.
    
    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
'''

# Standard library
import itertools

# External dependencies
import numpy as np
import cv2 as cv

# Project-specific
#import database
import utility
from generic import Detector

from config import DEBUG_LEVEL


class BoxExtractor(Detector):
    def __init__(self, variables):
        self.variables = variables

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
    def __init__(self, masks_dir, freq, threshold, default_shape, variables, buf_len=None):
        self.waiting_black = False
        super(Characters, self).__init__(masks_dir, freq, threshold, default_shape, variables, buf_len)

    def detect(self, frame, cur_count):
        height, width, _ = frame.shape
        focus_region = frame[np.ceil(height * 0.25) : np.ceil(height * 0.95),
                        np.ceil(width * 0.25) : np.ceil(width * 0.75)]
        if self.variables['is_black']:
            if self.waiting_black:
                self.store_players(focus_region)
            else:
                return
        if not self.variables['is_started'] and (cur_count % self.freq == 0):
            self.process(focus_region, cur_count, 0)
            if DEBUG_LEVEL > 1:
                cv.imshow(self.name(), focus_region)
                cv.waitKey(1)

    def process(self, frame, cur_count, player):
        """ Compares pre-loaded masks to current frame"""
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
        self.waiting_black = True
        timestamp = cur_count / self.variables['frame_rate']
        idx = self.buffer.exists(mask[1])
        if idx != -1:
            self.buffer[idx] = (mask[1], location, timestamp)
        else:
            self.buffer.append((mask[1], location, timestamp))

    def store_players(self, frame):
        self.waiting_black = False
        self.deactivate()
        ordered = list()
        max_place = 0
        characters = utility.find_unique(self.buffer, 0)
<<<<<<< HEAD
        ordered = self.sort_characters(characters)
        chars = [image[0].rsplit(".", 1)[0] for image in ordered]
        self.variables['characters'] = chars
        semi_ordered = self.sort_characters(characters, frame)
        print semi_ordered
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
        # Lock the players in
        chars = [player[0].split('.')[0] for player in ordered]
        if DEBUG_LEVEL > 1:
            for p, ch in enumerate(chars):
                print "Player %i is %s" % ((p+1), ch)
        self.variables['num_players'] = len(ordered)

    def sort_characters(self, characters, frame):
        """Sorting algorithm that ranks based on quadrant"""
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


#TODO Fix map masks!!!!
class Map(Detector):
    """Determines which map is being played (phase_0)"""
    def handle(self, frame, player, mask, cur_count, location):
        # Disable the detector and update race variables
        self.deactivate()
        self.variables['map'] = mask[1].split('.')[0]
        if DEBUG_LEVEL > 0:
            print '[%s]: Map is: %s' % (self.name(), mask[1].split('.')[0])
            cv.waitKey()


class StartRace(Detector):
    """Handles the beginning of a race in phase_0"""
    def handle(self, frame, player, mask, cur_count, location):
            self.variables['is_started'] = True
            self.deactivate()
            self.activate('EndRace')
            self.deactivate('Characters')
            self.deactivate('BoxExtractor')
            self.deactivate('Map')
            # Lock in player boxes (should be sorted alreadY)
            self.variables['player_regions'] = self.variables['player_regions']
            # Populate dictionary with start time
            self.variables['start_time'] = np.floor(cur_count / self.variables['frame_rate']) - 2
            if DEBUG_LEVEL > 0:
                print '[%s]: Race started at %d seconds' % (self.name(), self.variables['start_time'])
                cv.waitKey(1)


class EndRace(Detector):
    """Handles the end of a race (phase_0)"""
    def __init__(self, variables, session_id):
        self.variables = variables
        self.session_id = session_id

    def detect(self, frame, cur_count):
        if self.variables['is_started']:
            if self.variables['is_black']:
                # Either rage-quit or clean race finish (we'll handle rage quits later)
                self.handle(cur_count)
        else:
            self.deactivate()

    def handle(self, cur_count):
        self.variables['is_started'] = False
        # Populate dictionary with race duration
        self.variables['duration'] = np.ceil((cur_count / self.variables['frame_rate']) - self.variables['start_time'])
        self.variables['events'].append([(self.variables['start_time'], self.variables['duration'])])
        # On end race, deactivate EndRaceDetector, activate StartRaceDetector and CharDetector
        self.deactivate()
        self.activate('StartRace')
        self.activate('Characters')
        self.activate('Map')
        # Store as event for splitting
        self.create_event(start_time=self.variables['start_time'],
                          duration=self.variables['duration'])
        if DEBUG_LEVEL == 0:
            try:
                database.put_race(self.variables)
            except: #TODO Figure out exact exceptions
                # UH OH!
                pass
        else:
            print "[%s] End of race detected at t=%2.2f seconds" % (self.name(), self.variables['duration'])

