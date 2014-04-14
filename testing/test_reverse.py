import unittest
import cv2 as cv
import numpy as np

import detection
from test_utils import *

class ReverseTest(unittest.TestCase):
    def setUp(self):
        self.detector = detection.Reverse(masks_dir='./masks/reverse/',
                                        freq=1,
                                        threshold=0.05,
                                        default_shape=[(237, 306, 3)])

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

    def test_reverse(self):
        rv = video_loop('./testing/reverse/t1/t1.mov', self.detector, [(11, 316), (244, 477)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Reverse')
            self.assertEqual(event['event_subtype'], 'Reverse')
            self.assertEqual(event['info'], 'ReverseStart')

    def test_FailCase(self):
        rv = video_loop('./testing/reverse/FailCase/FailCase.mov', self.detector, [(11, 316), (244, 477)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 0)


if __name__ == '__main__':
    unittest.main()
