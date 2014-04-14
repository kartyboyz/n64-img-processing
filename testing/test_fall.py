import unittest
import cv2 as cv
import numpy as np

import detection
from test_utils import *

class FallTest(unittest.TestCase):
    def setUp(self):
        self.detector = detection.Fall(masks_dir='./masks/fall/',
                                    freq=1,
                                    threshold=0.02,
                                    default_shape=[(237, 305, 3)])
        self.variables = {
            'events' : list(),
            'lap' : 1,
            'place' : -1,
            'is_started' : False,
            'frame_rate' : float(30)
        }
        self.states = { 'Fall' : True }
        self.detector.set_variables(self.variables)
        self.detector.set_states(self.states)

    def test_start_s3(self):
        rv = video_loop('./testing/fall/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Fall')
            self.assertEqual(event['event_subtype'], 'Fall')
            self.assertEqual(event['info'], '')

    def test_start_emulator(self):
        rv = video_loop('./testing/fall/t2/t2.mov', self.detector, [(320, 638), (242, 475)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Fall')
            self.assertEqual(event['event_subtype'], 'Fall')
            self.assertEqual(event['info'], '')

    def test_start_corner(self):
        rv = video_loop('./testing/fall/t3/t3.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Fall')
            self.assertEqual(event['event_subtype'], 'Fall')
            self.assertEqual(event['info'], '')

    def test_start_FailCase(self):
        rv = video_loop('./testing/fall/FailCase/FailCase.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 0)

if __name__ == '__main__':
    unittest.main()
