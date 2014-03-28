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
            self.variables['player_boxes'][:] = []
            local = list()
            for coord in itertools.product(points[0], points[1]):
                local.append([(coord[0][0], coord[0][1]), (coord[1][0], coord[1][1])])
            self.variables['player_boxes'] = self.sort_boxes(local, cur_frame)
        else:
            # Completely black frame
            self.variables['player_boxes'] = [[(0, cur_frame.shape[1]), (0, cur_frame.shape[0])]]

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
        if self.variables['is_black']:
            if self.waiting_black:
                self.store_players()
            else:
                return
        if not self.variables['is_started'] and (cur_count % self.freq == 0):
            height, width, _ = frame.shape
            focus_region = frame[np.ceil(height * 0.25) : np.ceil(height * 0.95),
                                 np.ceil(width * 0.25) : np.ceil(width * 0.75)]
            self.process(focus_region, cur_count, 0)
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
            self.variables["p" + str(player + 1)] = char
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %d is %s!" % (self.name(), player + 1, char)
        self.variables['num_players'] = len(ordered)

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
            self.variables['player_boxes'] = self.variables['player_boxes']
            # Populate dictionary with start time
            self.variables['start_time'] = np.floor(cur_count / self.variables['frame_rate']) - 2

            events = self.variables['events']
            events.append('startsracs!')
            self.variables['events'] = events
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
            self.detector_states[self.name()] = False

    def handle(self, cur_count):
        self.variables['is_started'] = False
        # Populate dictionary with race duration
        self.variables['duration'] = np.ceil((cur_count / self.variables['frame_rate']) - self.variables['start_time'])
        self.variables['events'].append([(self.variables['start_time'], self.variables['duration'])])
        print self.variables['events']

        # On end race, deactivate EndRaceDetector, activate StartRaceDetector and CharDetector
        self.deactivate()
        self.activate('StartRace')
        self.activate('Characters')
        self.activate('Map')
        events = self.variables['events']
        events.append('endrace!')
        self.variables['events'] = events
        if DEBUG_LEVEL == 0:
            try:
                database.put_race(self.variables)
            except: #TODO Figure out exact exceptions
                # UH OH!
                pass
        else:
            print self.detector_states
            print "[%s] End of race detected at t=%2.2f seconds" % (self.name(), self.variables['duration'])

