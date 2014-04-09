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
    def __init__(self):
        self.past_timestamp = 0.0 # To be used for debouncing events

    def detect(self, frame, cur_count, player):
        if self.variables['is_started']:
            self.process(frame, cur_count, player)

    def process(self, frame, cur_count, player):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5,5), 1)
        _, gray = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
        black_count = float(np.sum(gray)) / float(gray.size)
        # If at least 80% of the frame is true black, race has stopped
        if black_count <= float(16):
            self.handle(frame, player, cur_count)
        if DEBUG_LEVEL > 1:
            cv.imshow('thresh', gray)
            cv.waitKey(1)

    def handle(self, frame, player, cur_count):
        # Create event
        timestamp = cur_count / self.variables['frame_rate']
        # First timestamp?
        if not self.past_timestamp:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                            event_subtype=self.name(),
                            timestamp=np.floor(timestamp),
                            player=player,
                            lap=self.variables['lap'],
                            place=self.variables['place'],
                            info="In shortcut cave")
            if DEBUG_LEVEL > 0:
                print "[%s]: Shortcut detected at %s seconds" % (self.name(), timestamp)
        # Does it meet the specifications for debouncing?
        elif (timestamp - self.past_timestamp) > 10.0:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                            event_subtype=self.name(),
                            timestamp=np.floor(timestamp),
                            player=player,
                            lap=self.variables['lap'],
                            place=self.variables['place'],
                            info="In shortcut cave")
            if DEBUG_LEVEL > 0:
                print "[%s]: Shortcut detected at %s seconds" % (self.name(), timestamp)



class FinishRace(Detector):
    """Detects the end of a race (phase_1)"""
    def process(self, frame, cur_count, player):
        best_val = 1
        best_mask = None
        if frame != None:
            # Convert to HSV and then threshold in range for yellow
            binary = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
            binary = cv.inRange(binary, (8, 185, 212), (40, 255, 255))
            # Blur again to smooth out thresholded frame
            binary = cv.GaussianBlur(binary, (5, 5), 1)
            if len(self.default_shape) != 1:
                for mask, shape in zip(self.masks, self.default_shape):
                    if frame.shape != shape:
                        scaled_mask = (cv.cvtColor(utility.scaleImage(frame, mask[0], shape), cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
                    if DEBUG_LEVEL > 1:
                        cv.imshow('thresh', binary)
                        cv.waitKey(1)
            else:
                for mask in self.masks:
                    if frame.shape != self.default_shape[0]:
                        scaled_mask = (cv.cvtColor(utility.scaleImage(frame, mask[0], self.default_shape[0]), cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
                    if DEBUG_LEVEL > 1:
                        cv.imshow('thresh', binary)
                        cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        # Deactivate the detector & PositionChange to be safe, and set the final place state variable
        self.deactivate()
        self.deactivate('PositionChange')
        self.variables['place'] = int(mask[1].split('_')[0])
        timestamp = cur_count / self.variables['frame_rate']
        self.create_event(event_type='Laps',
                          event_subtype=self.name(),
                          timestamp=np.floor(timestamp),
                          player=player,
                          lap=self.variables['lap'],
                          place=self.variables['place'],
                          info="Player %s finishing place: %s" % (player, mask[1].split('_')[0]))
        if DEBUG_LEVEL > 0:
            print "[%s]: Player %s finished in place: %s" % (self.name(), player, mask[1].split('_')[0])


class PositionChange(Detector):
    """Detector for handling changes in position/place"""
    def process(self, frame, cur_count, player):
        best_val = 1
        best_mask = None
        if frame != None:
            # Convert to HSV, threshold in range for yellow, and blur again.
            hsv = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
            binary = cv.inRange(hsv, (8, 185, 212), (40, 255, 255))
            binary = cv.GaussianBlur(binary, (5,5), 1)
            if len(self.default_shape) != 1:
                for mask, shape in zip(self.masks, self.default_shape):
                    if frame.shape != shape:
                        scaled_mask = (cv.cvtColor(utility.scaleImage(frame,mask[0], shape), 
                            cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 1:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
                    if DEBUG_LEVEL > 1:
                        cv.imshow('thresh', binary)
                        cv.waitKey(1)
            else:
                for mask in self.masks:
                    if frame.shape != self.default_shape[0]:
                            scaled_mask = (cv.cvtColor(utility.scaleImage(frame,mask[0], self.default_shape[0]), 
                                cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 1:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
                    if DEBUG_LEVEL > 1:
                        cv.imshow('thresh', binary)
                        cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        # Append the mask '#_place.png' to the ring buffer
        self.buffer.append(mask[1])
        # If this is the first place that is given, store it
        timestamp = cur_count / self.variables['frame_rate']
        if len(self.buffer) == 1:
            # Update place state variable and create an event
            self.variables['place'] = int(mask[1].split('_')[0])
            self.create_event(event_type=self.name(),
                              event_subtype='Initial',
                              timestamp=np.floor(timestamp),
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info=0)
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s place: %s" % (self.name(), player, self.buffer[len(self.buffer) - 1])
        # Check if the found mask is different than the previous one
        elif mask[1].split('_')[0] != self.buffer[len(self.buffer) - 2].split('_')[0]:
            # Update place state variable and create an event
            self.variables['place'] = int(mask[1].split('_')[0])
            if int(mask[1].split('_')[0]) > int(self.buffer[len(self.buffer) - 2].split('_')[0]):
                subtype = 'Passed'
            else:
                subtype = 'Passing'
            self.create_event(event_type=self.name(),
                              event_subtype=subtype,
                              timestamp=timestamp,
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info=int(self.buffer[0].split('_')[0])
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s went from %s place to %s place " % (self.name(), player, 
                    self.buffer[len(self.buffer) - 2].split('_')[0], self.buffer[len(self.buffer) - 1].split('_')[0])


class Lap(Detector):
    """Detector for lap changes"""
    def handle(self, frame, player, mask, cur_count, location):
        # Disregard if a detection on lap 3 occured on a lap 2 change or lap 2 on a lap 3 change
        if ((self.variables['lap'] == 1) and ('2' in mask[1])) or \
            ((self.variables['lap'] == 2) and ('3' in mask[1])):
            timestamp = cur_count / self.variables['frame_rate']
            if not self.past_timestamp:
                self.past_timestamp = timestamp
                # Increment the lap state variable and create an event
                self.variables['lap'] += 1
                self.create_event(event_type='Lap',
                                event_subtype='Lap Change',
                                timestamp=np.floor(timestamp),
                                player=player,
                                lap=self.variables['lap'],
                                place=self.variables['place'],
                                info="Player is now on lap %s" % (self.variables['lap']))
                if DEBUG_LEVEL > 0:
                    print "[%s]: Player %s is now on lap %d" % (self.name(), player, self.variables['lap'])
            # Fastest lap ever is 2.39 seconds on Wario Stadium (SC)
            elif (timestamp - self.past_timestamp) > 2.38:
                self.past_timestamp = timestamp
                # Increment the lap state variable and create an event
                self.variables['lap'] += 1
                self.create_event(event_type='Lap',
                                event_subtype='Lap Change',
                                timestamp=np.floor(timestamp),
                                player=player,
                                lap=self.variables['lap'],
                                place=self.variables['place'],
                                info="Player is now on lap %s" % (self.variables['lap']))
                if DEBUG_LEVEL > 0:
                    print "[%s]: Player %s is now on lap %d" % (self.name(), player, self.variables['lap'])
        # Deactivate on lap 3 detection
        if self.variables['lap'] >= 3:
            self.deactivate()


class Items(Detector):
    """Detector for MK64 items"""
    def __init__(self, masks_dir, freq, threshold, default_shape, buf_len=None):
        self.prev_item = ''
        super(Items, self).__init__(masks_dir, freq, threshold, default_shape, buf_len)

    def handle(self, frame, player, mask, cur_count, location):
        blank = 'blank_box' # Name of image containing blank item box
        self.buffer.append(mask[1])
        cur_item = self.buffer[len(self.buffer) - 2]
        # If this detection was a blank box and the last was not, continue with checks
        if (len(self.buffer) > 1) and (blank in mask[1]) and (blank not in cur_item):
            timestamp = cur_count / self.variables['frame_rate']
            if not self.past_timestamp:
                # Create an event
                self.create_event(event_type=self.name(),
                                  event_subtype='Item Get',
                                  timestamp=np.floor(timestamp),
                                  player=player,
                                  lap=self.variables['lap'],
                                  place=self.variables['place'],
                                  info="Player received a %s" % (cur_item.split('.')[0]))
                self.prev_item = cur_item
                self.buffer.clear()
                if DEBUG_LEVEL > 0:
                    print "[%s]: Player %s has %s" % (self.name(), player, cur_item)

            # Has been more than a second since the last event
            elif (timestamp - self.past_timestamp) > 0.45:
                # Was the last item boo? If so, the item received can only be detected on use.
                # Therefore, it doesn't matter if every item in the buffer is the same besides the last element.
                if self.prev_item == 'boo.png':
                    print self.prev_item
                    # Create an event
                    self.create_event(event_type=self.name(),
                                      event_subtype='Item Get',
                                      timestamp=np.floor(timestamp),
                                      player=player,
                                      lap=self.variables['lap'],
                                      place=self.variables['place'],
                                      info="Player received a %s" % (cur_item.split('.')[0]))
                    self.buffer.clear()
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Player %s has %s" % (self.name(), player, cur_item)
                # Are all items besides the last item in the buffer equal
                elif self.buffer.count(cur_item) != (len(self.buffer) - 1):
                    # If the current item is a single mushroom and the previous item was triple mushroom, do nothing.
                    if not (cur_item == 'boost_1.png' and self.prev_item == 'boost_3.png'):
                        # Create an event
                        self.create_event(event_type=self.name(),
                                          event_subtype='Item Get',
                                          timestamp=np.floor(timestamp),
                                          player=player,
                                          lap=self.variables['lap'],
                                          place=self.variables['place'],
                                          info="Player received a %s" % (cur_item.split('.')[0]))
                        self.buffer.clear()
                        if DEBUG_LEVEL > 0:
                            print "[%s]: Player %s has %s" % (self.name(), player, cur_item)
                self.prev_item = cur_item
            self.past_timestamp = timestamp


class Fall(Detector):
    """A detector for whenever the player falls of the map"""
    def handle(self, frame, player, mask, cur_count, location):
        timestamp = cur_count / self.variables['frame_rate']
        if not self.past_timestamp:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                              event_subtype=self.name(),
                              timestamp=np.floor(timestamp),
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info="Player fell off the map")
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s fell off the map" % (self.name(), player)
        elif (timestamp - self.past_timestamp) > 8.0:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                              event_subtype=self.name(),
                              timestamp=np.floor(timestamp),
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info="Player fell off the map")
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s fell off the map" % (self.name(), player)


class Reverse(Detector):
    """A detector for whenever the player drives in reverse"""
    def handle(self, frame, player, mask, cur_count, location):
        timestamp = cur_count / self.variables['frame_rate']
        if not self.past_timestamp:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                              event_subtype=self.name(),
                              timestamp=np.floor(timestamp),
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info="Player fell off the map")
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s is going in reverse for some reason" % (self.name(), player)
        elif (timestamp - self.past_timestamp) > 8.0:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                              event_subtype=self.name(),
                              timestamp=np.floor(timestamp),
                              player=player,
                              lap=self.variables['lap'],
                              place=self.variables['place'],
                              info="Player fell off the map")
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s is going in reverse for some reason" % (self.name(), player)



class BeginRace(Detector):
    """Handles the beginning of a race in phase_1"""
    def handle(self, frame, player, mask, cur_count, location):
        # Set lap state variable to 1, disable the detector, and create the event
        self.variables['lap'] = 1
        self.variables['is_started'] = True
        self.deactivate()
        timestamp = cur_count / self.variables['frame_rate']
        self.create_event(event_type='Laps',
                          event_subtype=self.name(),
                          timestamp=np.floor(timestamp),
                          player=player,
                          lap=self.variables['lap'],
                          place=0,
                          info="Race has begun")
        if DEBUG_LEVEL > 0:
            print '[%s]: Race started at %d seconds' % (self.name(), timestamp)
