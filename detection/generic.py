"""
Modularized detection suite containing the base Detector class
    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""
# Standard library
import os

# External dependencies
import cv2 as cv
import cv2.cv as cv1
import numpy as np

# Project-specific
import parallel
import utility
import detection

from utility import log
from config import DEBUG_LEVEL

class Detector(object):
    """Super (and abstract) class for all detectors, written specifically for MK64 events."""
    def __init__(self, masks_dir, freq, threshold, default_shape, buf_len=None):
        if type(self) is Detector:
            raise Exception("<Detector> should be subclassed.")
        self.masks = [(cv.imread(masks_dir+name), name)
                      for name in os.listdir(masks_dir)]
        for ii in xrange(len(self.masks)):
            self.masks[ii] = (cv.GaussianBlur(self.masks[ii][0], (3, 3), 1), self.masks[ii][1])
        self.freq = freq
        self.threshold = threshold
        if len(default_shape) != 1 and len(default_shape) != len(self.masks):
            raise Exception("Default shape must be of length 1 or length of masks list.")
        self.default_shape = default_shape
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)
        self.detector_states = None #To be filled in by setter
        self.variables = None
        self.past_timestamp = 0.0 # To be used for debouncing events

    def name(self):
        """Returns the name of the class."""
        return self.__class__.__name__

    def is_active(self, detector_name=None):
        """Return the detector activity state."""
        if detector_name == None:
            dname = self.name()
        else:
            dname = detector_name
        return self.detector_states[dname]

    def activate(self, detector_name=None):
        """Activates the detector given/self."""
        if detector_name == None:
            dname = self.name()
        else:
            dname = detector_name
        self.detector_states[dname] = True

    def deactivate(self, detector_name=None):
        """Deactivates the detector given/self."""
        if detector_name == None:
            dname = self.name()
        else:
            dname = detector_name
        self.detector_states[dname] = False

    def set_race_events_list(self, race_events_list):
        """Setter function for self.race_events_list"""
        self.race_events_list = race_events_list

    def set_states(self, states):
        """Setter function for self.detector_states."""
        self.detector_states = states

    def set_variables(self, variables):
        """Setter function for self.variables."""
        self.variables = variables

    def create_event(self, **kwargs):
        """Append/Setter function for events."""
        events = self.variables['events']
        events.append(kwargs)
        self.variables['events'] = events

    def detect(self, frame, cur_count, player):
        """Determines whether and how to process current frame."""
        if cur_count % self.freq is 0:
            frame = cv.GaussianBlur(frame, (3, 3), 1)
            self.process(frame, cur_count, player)

    def process(self, frame, cur_count, player):
        """Compares pre-loaded masks to current frame."""
        best_val = 1
        best_mask = None
        if len(self.default_shape) != 1:
            frame_roi = self.constrain_roi(frame)
            for mask, shape in zip(self.masks, self.default_shape):
                if frame.shape != shape:
                    scaled_mask = (utility.scaleImage(frame,mask[0], shape), mask[1])
                else:
                    scaled_mask = mask
                distances = cv.matchTemplate(frame_roi, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold and minval < best_val:
                    best_val = minval
                    best_mask = scaled_mask

            if best_mask is not None:
                self.handle(frame, player, best_mask, cur_count, minloc)
                if DEBUG_LEVEL > 1:
                    log("[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val))
        else:
            frame_roi = self.constrain_roi(frame)
            for mask in self.masks:
                if frame.shape != self.default_shape[0]:
                    scaled_mask = (utility.scaleImage(frame,mask[0], self.default_shape[0]), mask[1])
                else:
                    scaled_mask = mask
                distances = cv.matchTemplate(frame_roi, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold and minval < best_val:
                    best_val = minval
                    best_mask = scaled_mask
            if best_mask is not None:
                self.handle(frame, player, best_mask, cur_count, minloc)
                if DEBUG_LEVEL > 1:
                    log("[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val))

    def handle(self, frame, player, mask, cur_count, location):
        """
        Detectors should be subclassed with their own handle() function.
        raises a NotImplementedError if the subclass does not implement it.
        """
        raise NotImplementedError

    def constrain_roi(self, frame):
        """
        Detectors should be subclassed with their own constrain_region() function.
        raises a NotImplementedError if the subclass does not implement it.
        """
        raise NotImplementedError


class BlackFrame(Detector):
    """Faux-detector for determining if frame is black."""
    def __init__(self):
        """Needs no instance variables. Overrides superclass method."""
        pass

    def detect(self, frame, cur_count):
        """
        Sets variables['is_black'] False, and pass control to process().
        Overrides superclass method.
        """
        self.variables['is_black'] = False
        self.process(frame, cur_count)

    def process(self, frame, cur_count):
        """Checks frame for number of black pixels. Overrides superclass method."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
        black_count = float(np.sum(gray)) / float(gray.size)
        # If at least 80% of the frame is true black, race has stopped
        if black_count <= 0.2:
            self.handle(frame, cur_count)

    def handle(self, frame, cur_count):
        """Perform checks and debounce. Overrides superclass method."""
        self.variables['is_black'] = True
        if DEBUG_LEVEL > 1:
            log("[%s]: Handled" % (self.name()))


class Engine():
    """Driving module that feeds Workers video frames."""
    def __init__(self, variables, video_source):
        self.name = video_source
        self.capture = cv.VideoCapture(video_source)
        self.ret, self.frame = self.capture.read()
        self.variables = variables
        self.barrier = None
        self.manager = None

        # Fix framerate for Workers
        self.framerate = self.capture.get(5)
        if self.framerate == 0:
            self.framerate = 30
        for var in variables:
            var['frame_rate'] = self.framerate

        #DEBUG
        self.toggle = 1
        log("[%s] initialization complete." % (self.__class__.__name__))

    def setup_processes(self, num, regions):
        """Generates child processes."""
        self.barrier = utility.Barrier(parties=(num+1))
        self.manager = parallel.ProcessManager(num, regions, self.frame, self.barrier, self.variables)

    def add_detectors(self, detect_list):
        """Appends new detectors to Workers, wrapping the ProcessManager."""
        if self.barrier is None:
            raise RuntimeError("You need to call setup_processes() first")
        try:
            if 'KoopaTroopaBeach' not in self.variables[0]['course']:
                # Find SHORTCUT and remove it
                for detector in detect_list:
                    if isinstance(detector, detection.Shortcut):
                        detect_list.remove(detector)
                        break
        except:
            # Assume phase 0
            pass

        self.manager.set_detectors(detect_list)
        self.manager.start_workers()

    def process(self):
        """Loops through video source, feeding child processes new data."""
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
                    if DEBUG_LEVEL >  2:
                        cv.imshow(self.name, self.frame)
                        frame_count += 1
                        key = cv.waitKey(self.toggle)
                        if key is 27:
                            raise StopIteration
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
                try:
                    # Handle rangequits in Phase 1
                    for rv in self.variables:
                        for event in rv['events']:
                            if event['event_subtype'] == "Finish":
                                return self.variables
                    return None
                except:
                    # Phase 0 -- no handling
                    return self.variables
            except:
                # Any other exception is bad!
                return None

    def clear_buffer(self, offset):
        """
        Cleans up the rest of the buffer.
        Needed so that Workers don't process same data many times.
        """
        self.manager.image[offset:] = 0

    def cleanup(self):
        """Frees memory, alerts child processes to finish."""
        log("[%s] Cleaning up" % (self.__class__.__name__))
        self.manager.close()
        self.capture.release()
        self.barrier.abort()
