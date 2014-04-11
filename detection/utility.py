"""
Collection of objects frequently used in processing and splitting.
    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""
from collections import deque
import multiprocessing

import numpy as np
import cv2 as cv

import multiprocessing as mp
import time


class BrokenBarrierError(Exception):
    """Class for exception handling."""
    pass


class Barrier(object):
    """multiprocessing.Barrier implementation imported from Python 3.3"""
    def __init__(self, parties, action=None, timeout=None, action_args=()):
        """Class constructor."""
        if parties <= 0:
            raise ValueError('parties must be greater than 0')
        self._parties = parties
        self._action = action
        self._action_args = action_args
        self._timeout = timeout
        self._lock = mp.RLock()
        self._counter = mp.Semaphore(parties-1)
        self._wait_sem = mp.Semaphore(0)
        self._broken = mp.Semaphore(0)

    def wait(self, timeout=None):
        """Implementation of a semaphore"""
        # When each thread enters the semaphore it tries to do a
        # non-blocking acquire on _counter.  Since the original value
        # of _counter was parties-1, the last thread to enter will
        # fail to acquire the semaphore.  This final thread is the
        # "control" thread, and it is responsible for waking the
        # threads which arrived before it, waiting for them to
        # respond, calling the action (if any) and resetting barrier.
        # wait() returns 0 from the control thread; from all other
        # threads it returns -1.
        if timeout is None:
            timeout = self._timeout
        with self._lock:
            try:
                if self._counter.acquire(timeout=0):
                    # we are not the control thread
                    self._lock.release()
                    try:
                        # - wait to be woken by control thread
                        if not self._wait_sem.acquire(timeout=timeout):
                            raise BrokenBarrierError
                        res = -1
                    finally:
                        self._counter.release()
                        self._lock.acquire()
                else:
                    # we are the control thread
                    # - release the early arrivers
                    for i in range(self._parties-1):
                        self._wait_sem.release()
                    # - wait for all early arrivers to wake up
                    for i in range(self._parties-1):
                        temp = self._counter.acquire(timeout=5)
                        assert temp
                    # - reset state of the barrier
                    for i in range(self._parties-1):
                        self._counter.release()
                    # - carry out action and return
                    if self._action is not None:
                        self._action(*self._action_args)
                    res = 0
            except:
                self.abort()
                raise
            return res

    def abort(self):
        """Fuction to release all semaphores."""
        with self._lock:
            if self.broken:
                return
            self._broken.release()
            # release any waiters
            for i in range(self._parties - 1):
                self._wait_sem.release()

    @property
    def broken(self):
        return not self._broken._semlock._is_zero()

    @property
    def parties(self):
        return self._parties

    @property
    def n_waiting(self):
        with self._lock:
            if self.broken:
                raise BrokenBarrierError
            return (self._parties - 1) - self._counter.get_value()


def pixelate(image, resolution):
    """
    Global function to average and downsize an image.
    Generates a new image with dimesions [resolution]x[resolution],
    with each cell containing the average color of its corresponding
    block in the 'image' parameter.
    """
    h, w, _ = image.shape
    block_size = (h/resolution, w/resolution)
    rv = np.zeros((resolution, resolution, 3), np.uint8)
    # Debug
    display = np.zeros((160, 160, 3), np.uint8)
    block = 160/resolution
    for r in xrange(resolution):
        for c in xrange(resolution):
            # Determine coordinates
            start_x = block_size[1]*c
            start_y = block_size[0]*r
            # Calculate average of current block
            avg_b, avg_g, avg_r, _ = cv.mean(image[start_y:start_y+block_size[0], start_x:start_x+block_size[1]])
            # Populate return matrix
            rv[r][c] = [avg_b, avg_g, avg_r]
            # Debug
            cv.rectangle(display, (block*c, block*r), (block*c+block, block*r+block), [avg_b, avg_g, avg_r], -1)
    return rv, display



def scaleImage(frame, mask, frame_shape_default):
    """
    Generates a new image mask with dimensions scaled to the size of the video frame.
    This works by calculating the percent into the frame both (x1,y1) and (x2,y2) occur.
    (x1,y1) is the top left corner of the mask and (x2,y2) is the bottom right corner of the mask.
    Then compute the new x y pairs based on the calculated percentages and pass results onto cv.resize.
    Resizing operation is done via bilinear interpolation.
    """

    # Get the dimensions of the frame and the shape of the mask
    h_frame, w_frame, _ = frame.shape
    h_mask, w_mask, _ = mask.shape
    coords = ((0,0),(w_mask,h_mask))

    x1_percentage = 0.0
    y1_percentage = 0.0
    x2_percentage = coords[1][0] / float(frame_shape_default[1]) * 100.0
    y2_percentage = coords[1][1] / float(frame_shape_default[0]) * 100.0

    newx1 = 0
    newy1 = 0
    newx2 = int(np.ceil((x2_percentage * float(w_frame)) / 100.0))
    newy2 = int(np.ceil((y2_percentage * float(h_frame)) / 100.0))

    # Resize the image
    fx = float(newx2) / float(w_mask)
    fy = float(newy2) / float(h_mask)
    scaled_image = cv.resize(mask, (newx2,newy2), fx, fy, cv.INTER_LINEAR)
    return scaled_image


def in_range(number, low, high):
    """Determines if a number is bounded by low, high."""
    return (low <= number and number <= high)


class RingBuffer(deque):
    """
    Deque-based ring-buffer implementation. Initialized to a max size,
    pops off front when full.
    """
    def __init__(self, max_size):
        """Class constructor."""
        deque.__init__(self)
        self.__max_size = max_size

    def append(self, x):
        """Appends x to the buffer."""
        if len(self) == self.__max_size:
            self.popleft()
        deque.append(self, x)

    def all_same(self):
        """Return True if all items in buffer are equal. Otherwise, return False."""
        if self.count(self[0]) is (self.__max_size):
            return True;
        else:
            return False;

    def in_buffer(self, item):
        """Return True if exists, False otherwise. Used for item detection"""
        for mask in self:
            if item in mask:
                return True
        return False

    def exists(self, item):
        """Determines if item exists in the buffer. If so, return the index. Else, return -1."""
        for ii in xrange(len(self)):
            if item == self[ii][0]:
                return ii
        return -1

    def __deepcopy__(self, memo):
        """A deepcopy implementation that works with multiprocessing."""
        cls = self.__class__
        res = cls.__new__(cls)
        res.__max_size = self.__max_size
        memo[id(self)] = res
        return res

    def tolist(self):
        """Returns itself casted as a list()."""
        return list(self)


def find_unique(container, index):
    """
    Collects all unique elements in a container, biased according to
    'index.' Note that it's not very generic: it stores the index as the
    key and the following index as the value
    """
    results = dict()
    if index is not None:
        for thing in container:
            if not results.has_key(thing[index]):
                results[thing[index]] = (thing[index + 1], thing[index + 2])
    return results
