""" Modularized detection suite containing the phase_1 classes.

    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""

# External dependencies
import numpy as np
import cv2 as cv

# Project-specific
import utility
from generic import Detector

from config import DEBUG_LEVEL


class Shortcut(Detector):
    """Faux-detector for determining if frame is black.
    Most of the functions are overriding the superclass.
    Updates race variables that race has stopped if above is true
    """
    def __init__(self, variables):
        self.variables = variables

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
        if DEBUG_LEVEL > 1:
            cv.imshow('thresh', gray)
            cv.waitKey(1)

    def handle(self, frame, player, cur_count):
        # Create event
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event(event_type=self.name(),
                          event_subtype=self.name(),
                          timestamp=timestamp,
                          player=player,
                          lap=self.variables['lap'],
                          place=self.variables['place'],
                          info="In shortcut cave")
        if DEBUG_LEVEL > 0:
            print "[%s]: Shortcut detected" % (self.name())



class FinishRace(Detector):
    """Detects the end of a race (phase_1)"""
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
            for mask, shape in zip(self.masks, self.default_shape):
                if frame.shape != shape:
                    scaled_mask = (cv.cvtColor(utility.scaleImage(frame, mask[0], shape), cv.COLOR_BGR2GRAY), mask[1])
                else:
                    scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold:
                    print "Found %s :-) ------> %s" % (scaled_mask[1], minval)
                    self.handle(frame, player, scaled_mask, cur_count, minloc)
                if DEBUG_LEVEL > 1:
                    cv.imshow('thresh', binary)
                    cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        # Deactivate the detector & PositionChange to be safe, and set the final place state variable
        self.deactivate()
        self.deactivate('PositionChange')
        self.variables['place'] = int(mask[1].split('_')[0])
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event(event_type='Laps',
                          event_subtype=self.name(),
                          timestamp=timestamp,
                          player=player,
                          lap=self.variables['lap'],
                          place=self.variables['place'],
                          info="Player %s finishing place: %s" % (player, mask[1].split('_')[0]))
        if DEBUG_LEVEL > 0:
            print "[%s]: Player %s finished in place: %s" % (self.name(), player, mask[1].split('_')[0])


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

            for mask, shape in zip(self.masks, self.default_shape):
                if frame.shape != shape:
                    scaled_mask = (cv.GaussianBlur(cv.cvtColor(utility.scaleImage(frame,mask[0], shape), 
                        cv.COLOR_BGR2GRAY), (5, 5), 1), mask[1])
                else:
                    scaled_mask = (cv.GaussianBlur(cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), (5, 5), 1), mask[1])
                distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                minval, _, minloc, _ = cv.minMaxLoc(distances)
                if minval <= self.threshold:
                    if DEBUG_LEVEL > 0:
                        print "Found %s :-) ------> %s" % (scaled_mask[1], minval)
                    self.handle(frame, player, scaled_mask, cur_count, minloc)
                if DEBUG_LEVEL > 1:
                    cv.imshow('binary', binary)
                    cv.imshow('mask', scaled_mask[0])
                    cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        # Append the mask '#_place.png' to the ring buffer
        self.buffer.append(mask[1])
        # If this is the first place that is given, store it
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        if len(self.buffer) == 1:
            # Update place state variable and create an event
            self.variables['place'] = int(mask[1].split('_')[0])
            self.create_event(event_type=self.name(),
                              event_subtype='Initial',
                              timestamp=timestamp,
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info="Player %s finishing place: %s" % (player, mask[1].split('_')[0]))
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s place: %s" % (self.name(), player, self.buffer[len(self.buffer) - 1])
        # Check if the found mask is different than the previous one
        elif mask[1].split('_')[0] != self.buffer[len(self.buffer) - 2].split('_')[0]:
            # Update place state variable and create an event
            self.variables['place'] = int(mask[1].split('_')[0])
            self.create_event(event_type=self.name(),
                              event_subtype='Place change',
                              timestamp=timestamp,
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info="Player changed place from %s to %s" % (self.buffer[0][1].split('_')[0], self.buffer[1][1].split('_')[0]))
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s went from %s place to %s place " % (self.name(), player, 
                    self.buffer[len(self.buffer) - 2].split('_')[0], self.buffer[len(self.buffer) - 1].split('_')[0])


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
        # Create an event
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        if 'banana' in mask[1]:
            subtype = 'banana'
        else:
            subtype = 'shell or bomb'
        self.create_event(event_type=self.name(),
                          event_subtype=subtype,
                          timestamp=timestamp,
                          player=player,
                          lap=self.variables['lap'],
                          place=self.variables['place'],
                          info="Player collided with an object")
        if DEBUG_LEVEL > 0:
            print "[%s]: Player %s was hit with an item or bomb-omb" % (self.name(), player)


class Laps(Detector):
    """Detector for lap changes"""
    def process(self, frame, cur_count, player):
        frame = cv.GaussianBlur(frame, (5, 5), 1)
        super(Laps, self).process(frame, cur_count, player)

    def handle(self, frame, player, mask, cur_count, location):
        # TODO/xxx: debounce hits
        # Increment the lap state variable and create an event
        self.variables['lap'] += 1
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event(event_type='Lap',
                          event_subtype='Lap Change',
                          timestamp=timestamp,
                          player=player,
                          lap=self.variables['lap'],
                          place=self.variables['place'],
                          info="Player is now on lap %s" % (mask[1].split('.')[0]))
        if DEBUG_LEVEL > 0:
            print "[%s]: Player %s is on %s" % (self.__class__.__name__, player, mask[1])


class Items(Detector):
    """Detector for MK64 items"""
    def handle(self, frame, player, mask, cur_count, location):
        blank = 'blank_box.png' # Name of image containing blank item box
        self.buffer.append(mask[1])
        last_item = self.buffer[len(self.buffer) - 2]
        # Sorry for the gross if-statemen :-(
        if len(self.buffer) > 1 and mask[1] == blank and last_item != blank:
            # Create an event
            timestamp = np.floor(cur_count / self.variables['frame_rate'])
            self.create_event(event_type=self.name(),
                              event_subtype='Item Get',
                              timestamp=timestamp,
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info="Player received a %s" % (last_item.split('.')[0]))
            self.buffer.clear()
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s has %s" % (self.name(), player, last_item)
                cv.imshow('frame', frame)


class BeginRace(Detector):
    """Handles the beginning of a race in phase_1"""
    def handle(self, frame, player, mask, cur_count, location):
        # Set lap state variable to 1, disable the detector, and create the event
        self.variables['lap'] = 1
        self.deactivate()
        timestamp = np.floor(cur_count / self.variables['frame_rate'])
        self.create_event(event_type='Laps',
                          event_subtype=self.name(),
                          timestamp=timestamp,
                          player=player,
                          lap=self.variables['lap'],
                          place=0,
                          info="Race has begun")
