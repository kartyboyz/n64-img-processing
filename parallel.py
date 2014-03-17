import ctypes
import multiprocessing

import cv2 as cv
import numpy as np

import config
import utility

from config import DEBUG_LEVEL


# Number of frames to be passed to subprocesses
#    We can vary this to change how much memory is being used
#    MEM_USAGE = BUFFER_LENGTH * sizeof(array_element) * frame.size
BUFFER_LENGTH = 200


class Worker(multiprocessing.Process):
    """Worker process containing detetors, shared memory, and event triggers"""
    def __init__(self, shared_memory, barrier, bounds, shape, event, lock, variables):
        #CLEAN Are all of these necessary?
        multiprocessing.Process.__init__(self)
        self.race_vars = variables
        self.shared = shared_memory
        self.barrier = barrier
        self.bounds = bounds
        self.shape = shape
        self.size = reduce(lambda x, y: x*y, shape)
        self.event = event
        self.lock = lock
        self.done = multiprocessing.Event()
        self.count = 1
        self.detectors = list()
        self.detector_states = None # Will be populated once Detectors are added
        if bounds is None:
            self.phase = 0
        else:
            self.phase = 1

    def set_detectors(self, detector_list, detector_states):
        """Wrapper for adding more detectors"""
        self.detector_states = detector_states
        for name in detector_list:
            self.detectors.append(name)

    def run(self):
        """Waits for & consumes frame buffer, then applies Detectors on each frame

        The frame buffer is a C-like byte array, populated by frame pixels. It contains
        at most BUFFER_LENGTH frames at offsets defined by frame.size = HEIGHT * WIDTH * DEPTH.
        This function accesses each frame individually by these offsets and runs Detector.detect()
        on each one.
        """
        while True:
            if self.done.is_set():
                break
            self.event.wait() # Blocking - trigger MUST be set
            buff = np.frombuffer(self.shared.get_obj(), dtype=ctypes.c_ubyte)
            for i in xrange(BUFFER_LENGTH):
                offset = i * self.size
                cur_el = buff[offset : offset + self.size]
                frame =cur_el.reshape(self.shape[0], self.shape[1], self.shape[2])
                #CLEAN This is poorly written and uglay
                if self.phase is 0:
                    if self.count is 1: # First , initialize to full size
                        self.bounds = [(0, self.shape[0]), (0, self.shape[1])]
                    else:
                        # Update our bounds from BoxExtractor
                        self.bounds = self.race_vars.player_boxes[0]
                region = frame[self.bounds[1][0] : self.bounds[1][1],
                               self.bounds[0][0] : self.bounds[0][1]]
                if DEBUG_LEVEL > 1:
                    # This is just for fancy visual "animation" :-p
                    dbg = region.copy()
                    cv.putText(dbg, "Processing %i" % (i), (10, 40),
                            cv.FONT_HERSHEY_SIMPLEX, 1,
                            (50,255, 50), 2, 1)
                    cv.imshow("[%s]" % self.name, dbg)
                    cv.waitKey(1)

                # NOTE: This section is our current bottleneck.
                # Unfortunately it's a pretty big one, and it's due to OpenCV's matchTemplate()
                for d in self.detectors:
                    if d.is_active():
                        if d.name() is 'BoxExtractor': #CLEAN This is gross
                            d.detect(frame, self.count)
                        else:
                            d.detect(region, self.count)
                self.count += 1

            if DEBUG_LEVEL > 0:
                print "[%s] Has processed %i buffered frames" % (self.name, BUFFER_LENGTH)
            self.event.clear()
            self.barrier.wait()
        print '[%s] Exiting' % self.name


class ProcessManager(object):
    """Handles subprocesses & their shared memory"""
    def __init__(self, num, regions, video_source, barrier, variables):
        if regions is None:
            # Assume it's a 'flag' for Phase 0, so just let it trickle down
            pass
        elif len(regions) != num:
            raise Exception("[%s] Assertion failed: # regions != length(regions)" \
                            % (self.__class__.__name__))
        self.barrier = barrier
        # Shared memory buffer setup
        self.shared = multiprocessing.Array(ctypes.c_ubyte,
                                            video_source.size*BUFFER_LENGTH)
        self.image = np.frombuffer(self.shared.get_obj(), dtype=ctypes.c_ubyte)

        # Object instantiation
        #CLEAN
        self.triggers = [multiprocessing.Event() for _ in xrange(4)]
        self.locks = [multiprocessing.Lock() for _ in xrange(4)]
        shape = video_source.shape
        self.workers = [Worker(shared_memory=self.shared,
                               barrier=barrier,
                               bounds=regions[i],
                               shape=shape,
                               event=self.triggers[i],
                               lock=self.locks[i],
                               variables=variables) for i in xrange(num)]

    def set_detectors(self, detect_list, detector_states):
        """Wrapper for Workers"""
        for worker in self.workers:
            worker.set_detectors(detect_list, detector_states)

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