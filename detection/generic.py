''' Modularized detection suite containing the base Detector class
    
    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
'''
# Standard library
import os

# External dependencies
import cv2 as cv
import cv2.cv as cv1
import numpy as np

# Project-specific
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
        if len(default_shape) != 1 and len(default_shape) != len(self.masks):
            print len(self.masks), len(default_shape)
            raise Exception("Default shape must be of length 1 or length of masks list.")
        self.default_shape = default_shape
        self.variables = variables[0]
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)
        self.detector_states = None #To be filled in by setter

    def name(self):
        return self.__class__.__name__

    def is_active(self, detector_name=None):
        if detector_name == None:
            dname = self.name()
        else:
            dname = detector_name
        return self.detector_states[dname]

    def activate(self, detector_name=None):
        if detector_name == None:
            dname = self.name()
        else:
            dname = detector_name
        self.detector_states[dname] = True

    def deactivate(self, detector_name=None):
        if detector_name == None:
            dname = self.name()
        else:
            dname = detector_name
        self.detector_states[dname] = False

    def set_race_events_list(self, race_events_list):
        self.race_events_list = race_events_list

    def set_states(self, states):
        self.detector_states = states

    def set_variables(self, variables):
        self.variables = variables

    def create_event(self, **kwargs):
        events = self.variables['events']
        events.append(kwargs)
        self.variables['events'] = events

    def detect(self, frame, cur_count, player):
        """ Determines whether and how to process current frame"""
        if cur_count % self.freq is 0:
            self.process(frame, cur_count, player)

    def process(self, frame, cur_count, player):
        """ Compares pre-loaded masks to current frame"""
        best_val = 1
        best_mask = None
        if len(self.default_shape) != 1:
            for mask, shape in zip(self.masks, self.default_shape):
                if frame.shape != shape:
                    scaled_mask = (utility.scaleImage(frame,mask[0], shape), mask[1])
                else:
                    scaled_mask = mask
                distances = cv.matchTemplate(frame, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold and minval < best_val:
                    best_val = minval
                    best_mask = scaled_mask
            player = 0 #TODO: Remove this shit
            if best_mask is not None:
                self.handle(frame, player, best_mask, cur_count, minloc)
                if DEBUG_LEVEL > 0:
                    print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
        else:
            for mask in self.masks:
                if frame.shape != self.default_shape[0]:
                    scaled_mask = (utility.scaleImage(frame,mask[0], self.default_shape[0]), mask[1])
                else:
                    scaled_mask = mask
                distances = cv.matchTemplate(frame, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold and minval < best_val:
                    best_val = minval
                    best_mask = scaled_mask
            player = 0 #TODO: Remove this shit
            if best_mask is not None:
                self.handle(frame, player, best_mask, cur_count, minloc)
                if DEBUG_LEVEL > 0:
                    print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)

    def handle(self, frame, player, mask, cur_count, location):
        # Detectors should be subclassed with their own handle() function
        raise NotImplementedError


class BlackFrame(Detector):
    """Faux-detector for determining if frame is black.
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
            print "[%s]: Handled" % (self.name())


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
                self.barrier.wait()
                self.cleanup()
                return self.variables
            except:
                # Any other exception is bad!
                return None

    def clear_buffer(self, offset):
        """Cleans up the rest of the buffer

        Needed so that Workers don't process same data many times
        """
        self.manager.image[offset:] = 0

    def cleanup(self):
        """Frees memory, alerts child processes to finish"""
        self.manager.close()
        self.capture.release()
        self.barrier.abort()
