import unittest
import cv2 as cv
import numpy as np

import detection
from test_utils import *

class LapTest(unittest.TestCase):
    def setUp(self):
        self.detector = detection.Lap(masks_dir='./masks/laps/',
                                    freq=1,
                                    threshold=0.08,
                                    default_shape=[(237,314,3)])

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

    def test_lap_s3(self):
        rv = video_loop('./testing/lap/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][0]['event_type'], 'Lap')
        self.assertEqual(rv['events'][0]['event_subtype'], 'New')
        self.assertEqual(rv['events'][0]['info'], '2')
        self.assertEqual(rv['events'][1]['event_type'], 'Lap')
        self.assertEqual(rv['events'][1]['event_subtype'], 'New')
        self.assertEqual(rv['events'][1]['info'], '3')

    def test_lap_emulator(self):
        rv = video_loop('./testing/lap/t2/t2.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][0]['event_type'], 'Lap')
        self.assertEqual(rv['events'][0]['event_subtype'], 'New')
        self.assertEqual(rv['events'][0]['info'], '2')
        self.assertEqual(rv['events'][1]['event_type'], 'Lap')
        self.assertEqual(rv['events'][1]['event_subtype'], 'New')
        self.assertEqual(rv['events'][1]['info'], '3')


    def test_reverse_corner(self):
        rv = video_loop('./testing/lap/t3/t3.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 2)
        self.assertEqual(rv['events'][0]['event_type'], 'Lap')
        self.assertEqual(rv['events'][0]['event_subtype'], 'New')
        self.assertEqual(rv['events'][0]['info'], '2')
        self.assertEqual(rv['events'][1]['event_type'], 'Lap')
        self.assertEqual(rv['events'][1]['event_subtype'], 'New')
        self.assertEqual(rv['events'][1]['info'], '3')

    def test_FailCase(self):
        rv = video_loop('./testing/lap/FailCase/FailCase.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 0)


if __name__ == '__main__':
    unittest.main()
