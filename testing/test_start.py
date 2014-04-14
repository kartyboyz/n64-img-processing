import unittest
import cv2 as cv
import numpy as np

import detection
from test_utils import *

class StartTest(unittest.TestCase):
    def setUp(self):
        self.detector = detection.BeginRace(masks_dir='./masks/start/',
                                            freq=1,
                                            threshold=0.16,
                                            default_shape=[(237, 314, 3), (237, 344, 3)])
        self.variables = {
            'events' : list(),
            'lap' : 1,
            'place' : -1,
            'is_started' : False,
            'frame_rate' : float(30)
        }
        self.states = { 'BeginRace' : True }
        self.detector.set_variables(self.variables)
        self.detector.set_states(self.states)

    def test_start_s3(self):
        rv = video_loop('./testing/start/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'BeginRace')
            self.assertEqual(event['info'], 'BeginRace')

    def test_start_emulator(self):
        rv = video_loop('./testing/start/t2/t2.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'BeginRace')
            self.assertEqual(event['info'], 'BeginRace')

    def test_start_FailCase(self):
        rv = video_loop('./testing/start/FailCase/FailCase.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 0)

if __name__ == '__main__':
    unittest.main()
