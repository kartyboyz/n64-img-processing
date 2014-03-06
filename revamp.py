""" Revamped detection suite for MK64 

    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""
# Standard library
import ctypes
import multiprocessing
import os
import sys

# External dependencies
import cv2 as cv
import cv2.cv as cv1
import numpy as np

# Project-specific
import config
import utility

DEBUG_LEVEL = 1
BUFFER_LENGTH = 200


class Worker(multiprocessing.Process):
    """Worker process containing detetors, shared memory, and event triggers"""
    def __init__(self, shared_memory, barrier, bounds, shape, event, lock, data):
        multiprocessing.Process.__init__(self)
        self.race_vars = config.Race()
        self.shared = shared_memory
        self.barrier = barrier
        self.bounds = bounds
        self.shape = shape
        self.event = event
        self.lock = lock
        self.data = data
        self.coords = None
        self.done = multiprocessing.Event()
        self.count = 1
        self.detectors = list()
        self.detector_states = dict()

    def set_detectors(self, detector_list):
        for name in detector_list:
            self.detectors.append(name)

    def run(self):
        """Waits to be fed a new frame, then processes it"""
        while True:
            if self.done.is_set():
                break
            self.event.wait() # Blocking - trigger MUST be set

            for buf_el in self.shared:
                print "Buf"
                buff = np.frombuffer(buf_el.get_obj(), dtype=ctypes.c_ubyte)
                frame =buff.reshape(self.shape[0], self.shape[1], self.shape[2])
                region = frame[self.bounds[0][0] : self.bounds[1][0],
                           self.bounds[0][1] : self.bounds[1][1]]
                for d in self.detectors:
                    d.detect(region, self.count)
                if DEBUG_LEVEL > 0:
                    cv.imshow(self.name, region)
                    cv.waitKey(1)
                self.count += 1

            self.event.clear()
            #self.barrier.wait()
        print '[%s] Exiting' % self.name


class Detector(object):
    """Parent class for all detectors, written specifically for MK64 events """
    def __init__(self, masks_dir, freq, threshold, default_shape, race_vars, states, buf_len=None):
        #if type(self) is Detector:
        #    raise Exception("<Detector> should be subclassed.")
        self.masks = [(cv.imread(masks_dir+name), name)
                      for name in os.listdir(masks_dir)]
        self.freq = freq
        self.threshold = threshold
        self.default_shape = default_shape
        self.race_vars = race_vars
        #TODO: Does states need to be passed in?
        self.states = states
        if buf_len:
            self.buffer = utility.RingBuffer(buf_len)

    def detect(self, frame, cur_count):
        """ Determines whether and how to process current frame """
        if cur_count % self.freq is 0:
            self.process(frame, cur_count)

    def process(self, frame, cur_count):
        """ Compares pre-loaded masks to current frame"""
        for mask in self.masks:
            if frame.shape != self.default_shape:
                scaled_mask = (utility.scaleImage(frame,
                                                  mask[0],
                                                  self.default_shape), mask[1])
            else:
                scaled_mask = (mask[0], mask[1])
            distance_map = cv.matchTemplate(frame, scaled_mask[0], cv.TM_SQDIFF_NORMED)
            minval, _, minloc, _ = cv.minMaxLoc(distance_map)
            if minval <= self.threshold:
                print "Found %s :-)" % (mask[1])


class ProcessManager(object):
    """Handles subprocesses & their shaired memory"""
    def __init__(self, num, regions, video_source, barrier):
        if len(regions) != num:
            raise Exception("[%s] Assertion failed" % (self.__class__.__name__))
        self.barrier = barrier
        # Shared memory buffer setup
        self.manager = multiprocessing.Manager()

        self.shared = [multiprocessing.Array(ctypes.c_ubyte, video_source.size)
                       for _ in xrange(BUFFER_LENGTH)]
        self.image = [np.frombuffer(self.shared[idx].get_obj(), dtype=ctypes.c_ubyte)
                      for idx in xrange(BUFFER_LENGTH)]

        # Object instantiation
        #TODO Clean this 
        self.data = self.manager.list([None, None, None, None])
        self.triggers = [multiprocessing.Event() for _ in xrange(4)]
        self.locks = [multiprocessing.Lock() for _ in xrange(4)]
        shape = video_source.shape
        self.workers = [Worker(shared_memory=self.shared,
                               barrier=barrier,
                               bounds=regions[i],
                               shape=shape,
                               event=self.triggers[i],
                               lock=self.locks[i],
                               data=self.data) for i in xrange(num)]

    def set_detectors(self, detect_list):
        for worker in self.workers:
            worker.set_detectors(detect_list)

    def start_workers(self):
        for worker in self.workers:
            worker.start()

    def detect(self):
        """Alerts workers that new data is available for detection"""
        for trigger in self.triggers:
            trigger.set()

    def close(self):
        """Orders all contained workers to stop their tasks"""
        for idx, worker in enumerate(self.workers):
            worker.done.set()
            self.triggers[idx].set()


class Engine():
    """Driving module that feeds Workers video frames"""
    def __init__(self, video_source):
        self.name = video_source
        self.capture = cv.VideoCapture(video_source)
        self.ret, self.frame = self.capture.read()

        self.race_vars = config.Race()
        self.race_vars.race['frame_rate'] = self.capture.get(cv1.CV_CAP_PROP_FPS)

        self.barrier = None
        self.manager = None

        # Debug
        self.toggle = 1
        print '[Engine]: initialization complete.'

    def setup_processes(self, num, regions):
        """Generates child processes"""
        self.barrier = utility.Barrier(parties=(num+1))
        self.manager = ProcessManager(num, regions, self.frame, self.barrier)

    def add_detectors(self, detect_list):
        """Appends new detectors, wrapping the ProcessManager"""
        if self.barrier is None:
            raise RuntimeError("You need to call setup_processes() first")
        self.manager.set_detectors(detect_list)
        self.manager.start_workers()
        #for detector in detect_list:
        #    self.detector_states[detector.__class__.__name__] = True

    def process(self):
        """Loops through video source, feeding child processes new data"""
        frame_count = 0
        while self.frame is not None:
            for i in xrange(BUFFER_LENGTH):
                self.manager.image[i][:] = self.frame.ravel()
                self.ret, self.frame = self.capture.read()
                if self.frame is None:
                    break
            self.manager.detect()

            self.barrier.wait()

            if DEBUG_LEVEL > 0:
                cv.imshow(self.name, self.frame)
                frame_count += 1
                key = cv.waitKey(self.toggle)
                if key is 27:
                    return
                elif key is 32:
                    self.toggle ^= 1

    def cleanup(self):
        """Frees all memory, alerts child processes to finish"""
        self.barrier.abort()
        self.capture.release()
        self.manager.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Please specify a video source."
        exit(-1)

    ITEMS = Detector(masks_dir='./high_res_masks/item_masks/',
                     freq=1,
                     threshold=0.16,
                     default_shape=(237, 314, 3),
                     race_vars=None,
                     states=None)

    ENGINE = Engine(sys.argv[1])
    ENGINE.setup_processes(1, [((0, 0), (237, 314))])
    ENGINE.add_detectors([ITEMS])
    ENGINE.process()
    ENGINE.cleanup()
