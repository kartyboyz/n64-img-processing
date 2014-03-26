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
    * Consider increasing buffer length for potential speedup?
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
    def __init__(self, masks_dir, freq, threshold, default_shape, variables, buf_len=None):
        if type(self) is Detector:
            raise Exception("<Detector> should be subclassed.")
        self.masks = [(cv.imread(masks_dir+name), name)
                      for name in os.listdir(masks_dir)]
        self.freq = freq
        self.threshold = threshold
        self.default_shape = default_shape
        self.variables = variables[0]
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)
        self.detector_states = None #To be filled in by setter

    def name(self):
        return self.__class__.__name__

    def is_active(self):
        return self.detector_states[self.name()]

    def activate(self):
        self.detector_states[self.name()] = True

    def deactivate(self):
        self.detector_states[self.name()] = False

    def set_race_events_list(self, race_events_list):
        self.race_events_list = race_events_list

    def set_states(self, states):
        self.detector_states = states

    def set_variables(self, variables):
        self.variables = variables

    def create_event(self, event_type, event_subtype, timestamp, player, lap, place, info):
        # Create the event and then add it to the events list
        event = {'lap': lap,
                'place': place,
                'player': player,
                'event_type': event_type,
                'event_subtype': event_subtype,
                'event_info': str(info),
                'timestamp': timestamp}
        self.race_events_list.append(event)

    def detect(self, frame, cur_count, player):
        """ Determines whether and how to process current frame"""
        if cur_count % self.freq is 0:
            self.process(frame, cur_count, player)

    def process(self, frame, cur_count, player):
        """ Compares pre-loaded masks to current frame"""
        best_val = 1
        best_mask = None
        for mask in self.masks:
            if frame.shape != self.default_shape:
                scaled_mask = (utility.scaleImage(frame,
                                                  mask[0],
                                                  self.default_shape), mask[1])
            else:
                scaled_mask = mask
            distances = cv.matchTemplate(frame, mask[0], cv.TM_SQDIFF_NORMED)
            minval, _, minloc, _ = cv.minMaxLoc(distances)
            if minval <= self.threshold and minval < best_val:
                best_val = minval
                best_mask = mask
        player = 0 #TODO: Remove this shit
        if best_mask is not None:
            self.handle(frame, player, best_mask, cur_count, minloc, player)
            if DEBUG_LEVEL > 1:
                print "Found %s :-) ------> %s" % (best_mask[1], best_val)

    def handle(self, frame, player, mask, cur_count, location):
        # Detectors should be subclassed with their own handle() function
        raise NotImplementedError


class BlackFrame(Detector):
    """Faux-detector for determining if frame is black

    Most of the functions are overriding the superclass.
    Updates race variables that race has stopped if above is true
    """
    def __init__(self, variables):
        self.variables = variables

    def detect(self, frame, cur_count):
        self.variables['is_black'] = False
        self.process(frame, cur_count)

    def process(self, frame, cur_count):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
        black_count = float(np.sum(gray)) / float(gray.size)
        # If at least 80% of the frame is true black, race has stopped
        if black_count <= 0.2:
            self.handle(frame, cur_count)

    def handle(self, frame, cur_count):
        self.variables['is_black'] = True
        if DEBUG_LEVEL > 1:
            print "[%s] Handled" % (self.name())


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

        if utility.in_range(len(points[0]), 1, 2) and \
           utility.in_range(len(points[1]), 1, 2):
            self.variables['player_boxes'][:] = []
            for coord in itertools.product(points[0], points[1]):
                self.variables['player_boxes'].append([(coord[0][0], coord[0][1]), (coord[1][0], coord[1][1])])
            self.variables['player_boxes'] = self.sort_boxes(self.variables['player_boxes'])
        else:
            # Completely black frame
            self.variables['player_boxes'].append([(0, cur_frame.shape[1]), (0, cur_frame.shape[0])])

    def sort_boxes(self, boxes):
        """Sorting algorithm that places priority on "top left" boxes"""
        ordered = list()
        upper = np.max(boxes).astype(float)
        for box in boxes:
            box_normed = np.divide(box, upper)
            rank = np.sum(100 * box_normed[1]) + np.sum(10 * box_normed[0])
            ordered.append((box, rank))
        ordered = sorted(ordered, key=lambda x:x[1])
        result = [el[0] for el in ordered]
        return result


class Shortcut(Detector):
    """Faux-detector for determining if frame is black.
    Most of the functions are overriding the superclass.
    Updates race variables that race has stopped if above is true
    """
    def __init__(self, variables, states):
        self.variables = variables
        self.detector_states = states

    def detect(self, frame, cur_count, player):
        if self.variables['is_started']:
            self.process(frame, cur_count, player)

    def process(self, frame, cur_count, player):
        # TODO/xxx: REMOVE THIS
        player = 0
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5,5), 1)
        _, gray = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
        black_count = float(np.sum(gray)) / float(gray.size)
        # If at least 80% of the frame is true black, race has stopped
        if black_count <= float(20):
            print black_count
            self.handle(frame, player, cur_count)
        if DEBUG_LEVEL > 0:
            cv.imshow('thresh', gray)
            cv.waitKey(1)

    def handle(self, frame, player, cur_count):
        print "[%s] Shortcut detected" % (self.__class__.__name__)
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event(self.name(), self.name(), timestamp, player, 
            self.variables['lap'], sef.variables['place'], 'In shortcut cave')



class FinishRace(Detector):
    """Detects the end of a race"""
    def process(self, frame, cur_count, player):
        player = 0
        # First smooth out the image with a Gaussian blur
        if frame != None:
            frame = cv.GaussianBlur(frame, (5, 5), 1)
            # Convert to HSV and then threshold in range for yellow
            binary = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
            binary = cv.inRange(binary, (8, 185, 212), (40, 255, 255))
            # Blur again to smooth out thresholded frame
            binary = cv.GaussianBlur(binary, (5, 5), 1)
            for mask in self.masks:
                if frame.shape != self.default_shape:
                    scaled_mask = (cv.cvtColor(utility.scaleImage(frame, mask[0], self.default_shape), cv.COLOR_BGR2GRAY), mask[1])
                else:
                    scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold:
                    print scaled_mask[1] + '--->' + str(minval)
                    self.handle(frame, player, scaled_mask[1], cur_count, minloc)
                if DEBUG_LEVEL > 0:
                    cv.imshow('thresh', binary)
                    #cv.imshow('mask', scaled_mask[0])
                    cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        # TODO/xxx: do something
        print "[%s]: Player %s got %s place" % (self.__class__.__name__, player, mask)
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event('Laps', self.name(), timestamp, player, self.variables['lap'], 
            self.variables['place'], 'In shortcut cave')


class PositionChange(Detector):
    """Detector for handling changes in position/place"""
    def process(self, frame, cur_count, player):
        player = 0
        # First smooth out the image with a Gaussian blur
        if frame != None:
            frame = cv.GaussianBlur(frame, (5, 5), 1)
            # Convert to HSV, threshold in range for yellow, and blur again.
            hsv = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
            binary = cv.inRange(hsv, (8, 185, 212), (40, 255, 255))
            binary = cv.GaussianBlur(binary, (5,5), 1)

            for mask in self.masks:
                if frame.shape != self.default_shape:
                    scaled_mask = (cv.GaussianBlur(cv.cvtColor(utility.scaleImage(frame,mask[0], self.default_shape), 
                        cv.COLOR_BGR2GRAY), (5, 5), 1), mask[1])
                else:
                    scaled_mask = (cv.GaussianBlur(cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), (5, 5), 1), mask[1])
                distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold:
                    #print scaled_mask[1] + '-->' + str(minval)
                    self.handle(frame, player, scaled_mask, cur_count, minloc)
                if DEBUG_LEVEL > 0:
                    cv.imshow('binary', binary)
                    cv.imshow('mask', scaled_mask[0])
                    cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        # Append the mask '#_place.png' to the ring buffer
        self.buffer.append(mask[1])
        # If this is the first place that is given, store it
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        if len(self.buffer) == 1:
            # Update state variables
            print 'First occurence!'
            print "[%s]: Player %s is in %s place" % (self.name(), player, self.buffer[len(self.buffer) - 1])
            self.create_event(self.name(), 'Initial', timestamp, player, self.variables['lap'], 
                self.variables['place'], 'First given place is %s' % (self.buffer[0][1].split('.')[0]))
        # Check if the found mask is different than the previous one
        elif mask.split('_')[0] != self.buffer[len(self.buffer) - 2].split('_')[0]:
            # Update state variables
            self.create_event(self.name(), 'Place change', timestamp, player, self.variables['lap'], self.variables['place'], 
                'Player changed place from %s to %s' % (self.buffer[0][1].split('.')[0], self.buffer[1][1].split('.')[0]))
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s went from %s place to %s place " % (self.name(), player, 
                    self.buffer[len(self.buffer) - 2][0][1], self.buffer[len(self.buffer) - 1][1])


class Collisions(Detector):
    """Detector for collisions
            Collisions are when you get hit by a green shell, red shell, blue shell,
            bomb-omb, or banana.
    """
    def process(self, frame, cur_count, player):
        frame = cv.GaussianBlur(frame, (3, 3), 1)
        super(Collisions, self).process(frame, cur_count, player)

    def handle(self, frame, player, mask, cur_count, location):
        # TODO/xxx: debounce hits
        # Update state variables
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        if 'banana' in mask[1]:
            subtype = 'banana'
        else:
            subtype = 'shell or bomb'
        self.create_event(self.name(), subtype, timestamp, self.variables['lap'], self.variables['place'], 
            'Player collided with an object')
        if DEBUG_LEVEL > 0:
            print "[%s]: Player %s was hit with an item or bomb-omb" % (self.name(), player)


class Laps(Detector):
    """Detector for lap changes"""
    def process(self, frame, cur_count, player):
        frame = cv.GaussianBlur(frame, (5, 5), 1)
        super(Laps, self).process(frame, cur_count, player)

    def handle(self, frame, player, mask, cur_count, location):
        # TODO/xxx: debounce hits
        # Update state variables
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event('Lap', 'Lap Change', timestamp, player, self.variables['lap'], self.variables['place'],
            'Player is now on %s' % (mask[1].split('.')[0]))
        if DEBUG_LEVEL > 0:
            print "[%s]: Player %s is on %s" % (self.__class__.__name__, player, mask[1])


class Items(Detector):
    """Detector for MK64 items"""
    def handle(self, frame, player, mask, cur_count, location):
        blank = 'blank_box.png' # Name of image containing blank item box
        self.buffer.append(mask[1])
        last_item = self.buffer[len(self.buffer) - 2]
        # Sorry for the gross if-statemen :-(
        if len(self.buffer) > 1 and mask[1] is blank and last_item is not blank:
            #TODO Update JSON here
            timestamp = np.floor(cur_count / self.variables['frame_rate'])
            self.create_event(self.name(), 'Item get', timestamp, player, self.variables['lap'],
                self.variables['place'], 'Player received a %s' % (last_item.split('.')[0]))
            self.buffer.clear()
            if DEBUG_LEVEL > 1:
                print "[%s] Player %s has %s" % (self.name(), player, last_item)
                cv.imshow('frame', frame)


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
            self.process(focus_region, cur_count, player)
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
                print "Player %d is %s!" % (player + 1, char)
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

class StartRace(Detector):
    """Handles the beginning of a race in phase_0"""
    def handle(self, frame, player, mask, cur_count, location):
            self.variables['is_started'] = True
            self.detector_states['StartRace'] = False
            self.detector_states['EndRace']   = True
            self.detector_states['Characters'] = False
            self.detector_states['BoxExtractor'] = False
            self.detector_states['Map'] = False
            # Lock in player boxes (should be sorted alreadY)
            self.variables['player_boxes'] = self.variables['player_boxes']
            # Populate dictionary with start time
            self.variables['start_time'] = np.floor(cur_count / self.variables['frame_rate']) - 6
            if DEBUG_LEVEL > 0:
                print '[StartRace]: Race started at ' + str(self.variables['start_time']) + ' seconds.'


class BeginRace(Detector):
    """Handles the beginning of a race in phase_1"""
    def handle(self, frame, player, mask, cur_count, location):
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event('Laps', self.name(), timestamp, player, self.variables['lap'],
            self.variables['place'], 'Race has begun')

class EndRace(Detector):
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
        #TODO We could simplify this to make it cleaner
        # On end race, deactivate EndRaceDetector, activate StartRaceDetector and CharDetector
        self.detector_states['EndRace'] = False
        self.detector_states['StartRace'] = True
        self.detector_states['Characters'] = True
        self.detector_states['BoxExtractor'] = True
        self.detector_states['Map'] = True

        if DEBUG_LEVEL > 1:
            print self.detector_states
        # Populate dictionary with race duration
        self.variables['duration'] = np.ceil((cur_count / self.variables['frame_rate']) - self.variables['start_time'])
        if DEBUG_LEVEL == 0:
            try:
                database.put_race(self.variables)
            except: #TODO Figure out exact exceptions
                # Database error, dump to filesystem
                self.variables.save("dbfail_session%i.dump" % (self.variables['session_id']))
                #TODO Decide if we ever want to be able to recover dump files
        else:
            print "[%s] End of race detected at t=%2.2f seconds" % (self.name(), self.variables['duration'])

#TODO Fix map masks!!!!
class Map(Detector):
    """Determines which map is being played (phase_0)"""
    def handle(self, frame, player, mask, cur_count, location):
        self.detector_states[self.name()] = False
        if DEBUG_LEVEL > 0:
            print 'Map is: ' + mask[1]
            cv.waitKey()
        # Update JSON


class Engine():
    """Driving module that feeds Workers video frames"""
    def __init__(self, variables, video_source):
        self.name = video_source
        self.capture = cv.VideoCapture(video_source)
        self.ret, self.frame = self.capture.read()

        for _ in range(len(variables)):
            variables[0]['frame_rate'] = self.capture.get(cv1.CV_CAP_PROP_FPS)
        self.variables = variables
        self.barrier = None
        self.manager = None

        #DEBUG
        self.toggle = 1
        print "[%s] initialization complete." % (self.__class__.__name__)

    def setup_processes(self, num, regions):
        """Generates child processes"""
        self.barrier = utility.Barrier(parties=(num+1))
        self.manager = parallel.ProcessManager(num, regions, self.frame, self.barrier, self.variables)

    def add_detectors(self, detect_list):
        """Appends new detectors to Workers, wrapping the ProcessManager"""
        if self.barrier is None:
            raise RuntimeError("You need to call setup_processes() first")
        self.manager.set_detectors(detect_list)
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
