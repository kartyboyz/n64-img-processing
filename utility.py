from collections import deque
import multiprocessing

import numpy as np
import cv2 as cv

import multiprocessing as mp
import time


class BrokenBarrierError(Exception):
    pass


class Barrier(object):
    '''
    multiprocessing.Barrier implementation imported from Python 3.3
    '''
    def __init__(self, parties, action=None, timeout=None, action_args=()):
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
    '''
    Global function to average and downsize an image
    Generates a new image with dimesions [resolution]x[resolution],
    with each cell containing the average color of its corresponding
    block in the 'image' parameter
    '''
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


def resize(frame, mask, frame_shape_default):
    default = tuple([float(cast_me) for cast_me in iter(frame_shape_default)])
    ratios = np.divide(default, frame.shape)
    new_size = np.multiply(mask.shape, ratios).astype(int)
    scaled = cv.resize(mask, (new_size[0], new_size[1]))
    return scaled

def in_range(number, low, high):
    """Determines if a number is bounded by low, high"""
    return (low <= number and number <= high)


class RingBuffer(deque):
    '''
    Deque-based ring-buffer implementation. Initialized to a max size,
    pops off front when full.
    '''
    def __init__(self, max_size):
        deque.__init__(self)
        self.__max_size = max_size

    def append(self, x):
        deque.append(self, x)
        if len(self) == self.__max_size:
            self.popleft()

    def all_same(self):
        if self.count(self[0]) is (self.__max_size-1):
            return True;
        else:
            return False;

    def exists(self, item):
        if self.count(item) > 0:
            return True
        else:
            return False

    def tolist(self):
        return list(self)


def find_unique(container, index=None):
    '''
    Generic function for determining all unique elements in a container.
    If 'index' is specified, it compares that index of each element.
    '''
    results = list()
    if index is not None:
        for thing in container:
            if thing[index] not in results:
                results.append(thing)
    else:
        for thing in container:
            if thing not in results:
                results.append(thing)
    return results
