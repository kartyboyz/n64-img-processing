import unittest
import cv2 as cv
import numpy as np

import detection
from test_utils import *

class PositionChangeTest(unittest.TestCase):
    def setUp(self):
        self.detector = detection.PositionChange(masks_dir='./masks/position/',
                                                freq=1,
                                                threshold=0.16,
                                                default_shape=[(237,314,3)],
                                                buf_len=2)
        self.variables = {
            'events' : list(),
            'lap' : 1,
            'place' : -1,
            'is_started' : False,
            'frame_rate' : float(30)
        }
        self.states = { self.detector.name() : True }
        self.detector.set_variables(self.variables)
        self.detector.set_states(self.states)

    def test_toFirst_s3(self):
        rv = video_loop('./testing/position/to_1/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][1]['event_type'], 'PositionChange')
        self.assertEqual(rv['events'][1]['event_subtype'], 'Passing')
        self.assertEqual(rv['events'][1]['info'], '1')

    def test_toFirst_emulator(self):
        rv = video_loop('./testing/position/to_1/t2/t2.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][1]['event_type'], 'PositionChange')
        self.assertEqual(rv['events'][1]['event_subtype'], 'Passing')
        self.assertEqual(rv['events'][1]['info'], '1')

    def test_toSecond_s3(self):
        rv = video_loop('./testing/position/to_2/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][1]['event_type'], 'PositionChange')
        self.assertEqual(rv['events'][1]['event_subtype'], 'Passed')
        self.assertEqual(rv['events'][1]['info'], '2')

    def test_toSecond_emulator(self):
        rv = video_loop('./testing/position/to_2/t2/t2.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][1]['event_type'], 'PositionChange')
        self.assertEqual(rv['events'][1]['event_subtype'], 'Passed')
        self.assertEqual(rv['events'][1]['info'], '2')

    def test_toThird_s3(self):
        rv = video_loop('./testing/position/to_3/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][1]['event_type'], 'PositionChange')
        self.assertEqual(rv['events'][1]['event_subtype'], 'Passed')
        self.assertEqual(rv['events'][1]['info'], '3')

    def test_toThird_emulator(self):
        rv = video_loop('./testing/position/to_3/t2/t2.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][1]['event_type'], 'PositionChange')
        self.assertEqual(rv['events'][1]['event_subtype'], 'Passed')
        self.assertEqual(rv['events'][1]['info'], '3')

    def test_toFourth_emulator(self):
        rv = video_loop('./testing/position/to_4/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][1]['event_type'], 'PositionChange')
        self.assertEqual(rv['events'][1]['event_subtype'], 'Passed')
        self.assertEqual(rv['events'][1]['info'], '4')

if __name__ == '__main__':
    unittest.main()
