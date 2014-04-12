"""
Modularized detection suite containing the phase_1 classes.
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
    """
    Faux-detector for determining if frame is black.
    Most of the functions are overriding the superclass.
    Updates race variables that race has stopped if frame is black.
    """
    def __init__(self):
        """Class constructor. Overrides the superclass method."""
        self.past_timestamp = 0.0 # To be used for debouncing events

    def detect(self, frame, cur_count, player):
        """
        Determines whether and how to process current frame.
        Overrides superclass method.
        """
        if self.variables['is_started']:
            self.process(frame, cur_count, player)

    def process(self, frame, cur_count, player):
        """Checks frame for number of black pixels. Overrides superclass method."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5,5), 1)
        _, gray = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
        black_count = float(np.sum(gray)) / float(gray.size)
        # If at least 80% of the frame is true black, race has stopped
        if black_count <= float(16):
            self.handle(frame, player, cur_count)
        if DEBUG_LEVEL > 2:
            cv.imshow('thresh', gray)
            cv.waitKey(1)

    def handle(self, frame, player, cur_count):
        """Perform checks and debounce. Overrides superclass method."""
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
                            info="KoopaTroopaBeachCave")
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
                            info="KoopaTroopaBeachCave")
            if DEBUG_LEVEL > 0:
                print "[%s]: Shortcut detected at %s seconds" % (self.name(), timestamp)



class FinishRace(Detector):
    """Detects the end of a race."""
    def process(self, frame, cur_count, player):
        """Compares pre-loaded masks to current frame. Overrides superclass method."""
        best_val = 1
        best_mask = None
        if frame != None:
            # Convert to HSV and then threshold in range for yellow
            binary = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
            binary = cv.inRange(binary, (8, 185, 212), (40, 255, 255))
            # Blur again to smooth out thresholded frame
            binary = cv.GaussianBlur(binary, (5, 5), 1)
            if len(self.default_shape) != 1:
                binary_roi = self.constrain_roi(binary)
                for mask, shape in zip(self.masks, self.default_shape):
                    if frame.shape != shape:
                        scaled_mask = (cv.cvtColor(utility.scaleImage(frame, mask[0], shape), cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary_roi, scaled_mask[0], cv.TM_SQDIFF_NORMED)
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
                binary_roi = self.constrain_roi(binary)
                for mask in self.masks:
                    if frame.shape != self.default_shape[0]:
                        scaled_mask = (cv.cvtColor(utility.scaleImage(frame, mask[0], self.default_shape[0]), cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary_roi, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 0:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
                    if DEBUG_LEVEL > 2:
                        cv.imshow('thresh', binary)
                        cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and deactivate. Overrides superclass method."""
        # Deactivate the detector & PositionChange to be safe, and set the final place state variable
        self.deactivate()
        self.deactivate('PositionChange')
        self.variables['place'] = int(mask[1][0])
        timestamp = cur_count / self.variables['frame_rate']
        self.create_event(event_type='Lap',
                        event_subtype="Finish",
                        timestamp=np.floor(timestamp),
                        player=player,
                        lap=self.variables['lap'],
                        place=self.variables['place'],
                        info=mask[1][0])
        if DEBUG_LEVEL > 0:
            print "[%s]: Player %s finished in place: %s" % (self.name(), player, mask[1][0])

    def constrain_roi(self, frame):
        """Constrains frame w.r.t. FinishRace. Overrides superclass method."""
        h, _ = frame.shape
        frame = frame[np.ceil(h / 3):h, :]
        return frame


class PositionChange(Detector):
    """Detector for handling changes in position/place."""
    def process(self, frame, cur_count, player):
        """Compares pre-loaded masks to current frame. Overrides superclass method."""
        best_val = 1
        best_mask = None
        if frame != None:
            # Convert to HSV, threshold in range for yellow, and blur again.
            hsv = cv.cvtColor(frame , cv.COLOR_BGR2HSV)
            binary = cv.inRange(hsv, (8, 185, 212), (40, 255, 255))
            binary = cv.GaussianBlur(binary, (5,5), 1)
            if len(self.default_shape) != 1:
                binary_roi = self.constrain_roi(binary)
                for mask, shape in zip(self.masks, self.default_shape):
                    if frame.shape != shape:
                        scaled_mask = (cv.cvtColor(utility.scaleImage(frame,mask[0], shape), 
                            cv.COLOR_BGR2GRAY), mask[1])
                    else:
                        scaled_mask = (cv.cvtColor(mask[0], cv.COLOR_BGR2GRAY), mask[1])
                    distances = cv.matchTemplate(binary_roi, scaled_mask[0], cv.TM_SQDIFF_NORMED)
                    minval, _, minloc, _ = cv.minMaxLoc(distances)
                    if minval <= self.threshold and minval < best_val:
                        best_val = minval
                        best_mask = scaled_mask
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 1:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
                    if DEBUG_LEVEL > 1:
                        cv.imshow('thresh', binary_roi)
                        cv.waitKey(1)
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
                if best_mask is not None:
                    self.handle(frame, player, best_mask, cur_count, minloc)
                    if DEBUG_LEVEL > 1:
                        print "[%s]: Found %s :-) ------> %s" % (self.name(), best_mask[1], best_val)
                    if DEBUG_LEVEL > 2:
                        cv.imshow('thresh', binary_roi)
                        cv.waitKey(1)

    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and debounce. Overrides the superclass method."""
        # Append the mask '#_place.png' to the ring buffer
        self.buffer.append(mask[1][0])
        # If this is the first place that is given, store it
        timestamp = cur_count / self.variables['frame_rate']
        if len(self.buffer) == 1:
            # Update place state variable and create an event
            self.variables['place'] = int(mask[1][0])
            self.create_event(event_type=self.name(),
                            event_subtype='Initial',
                            timestamp=np.floor(timestamp),
                            player=player,
                            lap=self.variables['lap'],
                            place=self.variables['place'],
                            info=mask[1][0])
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s place: %s" % (self.name(), player, self.buffer[len(self.buffer) - 1])
        # Check if the found mask is different than the previous one
        elif mask[1][0] != self.buffer[len(self.buffer) - 2]:
            # Update place state variable and create an event
            self.variables['place'] = int(mask[1][0])
            if int(mask[1][0]) > int(self.buffer[len(self.buffer) - 2]):
                subtype = 'Passed'
            else:
                subtype = 'Passing'
            self.create_event(event_type=self.name(),
                            event_subtype=subtype,
                            timestamp=timestamp,
                            player=player,
                            lap=self.variables['lap'],
                            place=self.variables['place'],
                            info=mask[1][0])
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s went from %s place to %s place " % (self.name(), player, 
                    self.buffer[len(self.buffer) - 2], self.buffer[len(self.buffer) - 1])

    def constrain_roi(self, frame):
        """Constrains frame w.r.t. PositionChange. Overrides superclass method."""
        h, _ = frame.shape
        frame = frame[np.ceil(h * 0.5):h, :]
        return frame


class Lap(Detector):
    """Detector for lap changes."""
    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and debounce. Overrides superclass method."""
        # Disregard if a detection on lap 3 occured on a lap 2 change or lap 2 on a lap 3 change
        if ((self.variables['lap'] == 1) and ('2' in mask[1])) or \
            ((self.variables['lap'] == 2) and ('3' in mask[1])):
            timestamp = cur_count / self.variables['frame_rate']
            if not self.past_timestamp:
                self.past_timestamp = timestamp
                # Increment the lap state variable and create an event
                self.variables['lap'] += 1
                self.create_event(event_type='Lap',
                                event_subtype='Few',
                                timestamp=np.floor(timestamp),
                                player=player,
                                lap=self.variables['lap'],
                                place=self.variables['place'],
                                info=str(self.variables['lap']))
                if DEBUG_LEVEL > 0:
                    print "[%s]: Player %s is now on lap %d" % (self.name(), player, self.variables['lap'])
            # Fastest lap ever is 2.39 seconds on Wario Stadium (SC)
            elif (timestamp - self.past_timestamp) > 2.38:
                self.past_timestamp = timestamp
                # Increment the lap state variable and create an event
                self.variables['lap'] += 1
                self.create_event(event_type='Lap',
                                event_subtype='Few',
                                timestamp=np.floor(timestamp),
                                player=player,
                                lap=self.variables['lap'],
                                place=self.variables['place'],
                                info=str(self.variables['lap']))
                if DEBUG_LEVEL > 0:
                    print "[%s]: Player %s is now on lap %d" % (self.name(), player, self.variables['lap'])
        # Deactivate on lap 3 detection
        if self.variables['lap'] >= 3:
            self.deactivate()

    def constrain_roi(self, frame):
        """Constrains frame w.r.t. Lap. Overrides superclass method."""
        return frame


class Items(Detector):
    """Detector for MK64 items"""
    def __init__(self, masks_dir, freq, threshold, default_shape, buf_len=None):
        """Class constructor. Overrides superclass method."""
        self.item_hist = utility.RingBuffer(3) # Used to track item history
        self.blank_count = 0 # A mod 2 variable that will be incremented. i.e. = {0, 1}
        super(Items, self).__init__(masks_dir, freq, threshold, default_shape, buf_len)
    
    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and debounce. Overrides superclass method."""
        blank = 'BlankBox'
        # If the mask is not in the buffer, append it
        if not len(self.buffer) or mask[1] != self.buffer[len(self.buffer) - 1]:
            self.buffer.append(mask[1])

        # Case 1: buffer is longer than 1 and current element is blank_box
        if (len(self.buffer) > 1) and (blank in mask[1]):
            cur_item = self.buffer[len(self.buffer) - 2]
            timestamp = cur_count / self.variables['frame_rate'] # Update on blank_box detection
            # Check blank_count. If 0, create event. This implies received an item
            if not self.blank_count:
                # Create an event
                self.create_event(event_type=self.name(),
                                event_subtype='ItemGet',
                                timestamp=np.floor(timestamp),
                                player=player,
                                lap=self.variables['lap'],
                                place=self.variables['place'],
                                info=cur_item.split('.')[0])
                self.blank_count ^= 1 # Toggle
                self.item_hist.append(cur_item)
                if DEBUG_LEVEL > 0:
                    print "[%s]: Player %d has a %s" % (self.name(), player, cur_item.split('.')[0])
            # Already saw a blank_box. Been long enough since last blank_box
            elif self.blank_count and (timestamp - self.past_timestamp) > 0.45:
                # item_hist length > 1
                if len(self.item_hist) > 1:
                    # If boo is last item in item_hist, item was stolen
                    if self.item_hist[len(self.item_hist) - 1] == 'Boo.png':
                        # Must check if the stolen item was a triple boost
                        if 'TripleMushroom.png' in self.item_hist and 'SingleMushroom.png' in self.item_hist:
                            self.create_event(event_type=self.name(),
                                        event_subtype='ItemStolen',
                                        timestamp=np.floor(timestamp),
                                        player=player,
                                        lap=self.variables['lap'],
                                        place=self.variables['place'],
                                        info="TripleMushroom")
                            self.blank_count ^= 1
                            self.item_hist.clear()
                            if DEBUG_LEVEL > 0:
                                print "[%s]: Player %d was robbed of a triple mushroom" % (self.name(), player)
                        else:
                            if DEBUG_LEVEL > 0:
                                print "[%s]: Player %d was robbed of a %s" % \
                                    (self.name(), player, self.item_hist[len(self.item_hist) - 2])
                            self.create_event(event_type=self.name(),
                                            event_subtype='ItemStolen',
                                            timestamp=np.floor(timestamp),
                                            player=player,
                                            lap=self.variables['lap'],
                                            place=self.variables['place'],
                                            info=self.item_hist[len(self.item_hist) - 2].split('.')[0])
                            self.blank_count ^= 1
                            self.item_hist.clear()
                    # If boo is the first item in item_hist, an item was stolen
                    elif self.item_hist[0] == 'Boo.png':
                        if 'TripleMushroom.png' in self.item_hist and 'SingleMushroom.png' in self.item_hist:
                            self.create_event(event_type=self.name(),
                                            event_subtype='ItemGet',
                                            timestamp=np.floor(timestamp),
                                            player=player,
                                            lap=self.variables['lap'],
                                            place=self.variables['place'],
                                            info="TripleMushroom")
                            self.blank_count ^= 1
                            self.item_hist.clear()
                            if DEBUG_LEVEL > 0:
                                print "[%s]: Player %d received a triple mushroom" % (self.name(), player)
                    # Must check if we got a triple boost
                    elif self.item_hist[len(self.item_hist) - 1] == 'SingleMushroom.png' and \
                        self.item_hist[len(self.item_hist) - 2] == 'TripleMushroom.png':
                        self.item_hist.clear()
                        self.blank_count ^= 1
                    # We've received a blank box, that meets timeouts criteria
                    # The last item was not a boo or a boost_3. Therefore, it must be an event
                else:
                    self.create_event(event_type=self.name(),
                                    event_subtype='ItemGet',
                                    timestamp=np.floor(timestamp),
                                    player=player,
                                    lap=self.variables['lap'],
                                    place=self.variables['place'],
                                    info=self.item_hist[len(self.item_hist) - 1])
                    self.blank_count ^= 1
                    if DEBUG_LEVEL > 0:
                        print "[%s] Player %d received a %s" % \
                            (self.name(), player, self.item_hist[len(self.item_hist) - 1])
            self.past_timestamp = timestamp # update the past timestamp to reflect the blank_box detection
            self.buffer.clear()
        
        # Case 2: Already saw 
        elif self.blank_count and (blank not in mask[1]) and not self.item_hist.in_buffer(mask[1]):
            # If the item detected is Boo, and Boo not in item_hist, append it
            if mask[1] == 'Boo.png':
                self.item_hist.append(mask[1])
            # Is the item a TripleMushroom
            elif mask[1] == 'TripleMushroom.png':
                self.item_hist.append(mask[1])
            # If item detected is SingleMushroom and TripleMushroom in item_hist, append it
            elif mask[1] == 'SingleMushroom.png' and self.item_hist.in_buffer('TripleMushroom'):
                self.item_hist.append(mask[1])
            # If item found after BlankBox is not Boo, SingleMushroom, or same item, clear item_hist and toggle blank_count
            else:
                self.blank_count ^= 1
                self.item_hist.clear()
                self.buffer.clear()
        if DEBUG_LEVEL > 2:
            print self.blank_count, self.item_hist

    def constrain_roi(self, frame):
        """Constrains frame w.r.t. Items. Overrides superclass method."""
        h, w, _ = frame.shape
        frame = frame[0:np.ceil(h * 0.36), :]
        return frame


class Fall(Detector):
    """A detector for whenever the player falls of the map"""
    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and debounce. Overrides superclass method."""
        timestamp = cur_count / self.variables['frame_rate']
        if not self.past_timestamp:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                            event_subtype=self.name(),
                            timestamp=np.floor(timestamp),
                            player=player,
                            lap=self.variables['lap'],
                            place=self.variables['place'],
                            info="")
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
                            info="")
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s fell off the map" % (self.name(), player)
    def constrain_roi(self, frame):
        """Constrains frame w.r.t. Fall. Overrides superclass method."""
        h, w, _ = frame.shape
        frame = frame[:, np.ceil(w * 0.16):np.ceil(w * 0.8)]
        return frame



class Reverse(Detector):
    """A detector for whenever the player drives in reverse"""
    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and debounce. Overrides superclass method."""
        timestamp = cur_count / self.variables['frame_rate']
        if not self.past_timestamp:
            self.past_timestamp = timestamp
            self.create_event(event_type=self.name(),
                            event_subtype=self.name(),
                            timestamp=np.floor(timestamp),
                            player=player,
                            lap=self.variables['lap'],
                            place=self.variables['place'],
                            info="ReverseStart")
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
                            info="ReverseStart")
            if DEBUG_LEVEL > 0:
                print "[%s]: Player %s is going in reverse for some reason" % (self.name(), player)

    def constrain_roi(self, frame):
        """Constrains frame w.r.t. Reverse. Overrides superclass method."""
        h, w, _ = frame.shape
        frame = frame[0:np.ceil(h / 1.3), :]
        return frame


class BeginRace(Detector):
    """Handles the beginning of a race in phase_1"""
    def handle(self, frame, player, mask, cur_count, location):
        """Perform checks and debounce. Overrides superclass method."""
        # Set lap state variable to 1, disable the detector, and create the event
        self.variables['lap'] = 1
        self.variables['is_started'] = True
        self.deactivate()
        timestamp = cur_count / self.variables['frame_rate']
        self.create_event(event_type='Lap',
                        event_subtype=self.name(),
                        timestamp=np.floor(timestamp),
                        player=player,
                        lap=self.variables['lap'],
                        place=0,
                        info=self.name())
        if DEBUG_LEVEL > 0:
            print '[%s]: Race started at %d seconds' % (self.name(), timestamp)

    def constrain_roi(self, frame):
        """Constrains frame w.r.t. BeginRace/StartRace. Overrides superclass method."""
        h, w, _ = frame.shape
        frame = frame[0:np.ceil(h * 0.5), np.ceil(w * 0.4):np.ceil(w * 0.75)]
        return frame