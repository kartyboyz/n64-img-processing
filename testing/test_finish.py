import unittest
import cv2 as cv
import numpy as np

import detection
from test_utils import *

class FinishRaceTest(unittest.TestCase):
    def setUp(self):
        self.detector = detection.FinishRace(masks_dir='./masks/finish/',
                                    freq=1,
                                    threshold=0.16,
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

    def test_first_s3(self):
        rv = video_loop('./testing/finish/first/t1/t1.mov', self.detector, [(320, 638), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'Finish')
            self.assertEqual(event['info'], '1')

    def test_first_emulator(self):
        rv = video_loop('./testing/finish/first/t2/t2.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'Finish')
            self.assertEqual(event['info'], '1')

    def test_second_s3(self):
        rv = video_loop('./testing/finish/second/t1/t1.mov', self.detector, [(4, 317), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'Finish')
            self.assertEqual(event['info'], '2')

    def test_second_emulator(self):
        rv = video_loop('./testing/finish/second/t2/t2.mov', self.detector, [(320, 638), (242, 475)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'Finish')
            self.assertEqual(event['info'], '2')

    def test_third_s3(self):
        rv = video_loop('./testing/finish/third/t1/t1.mov', self.detector, [(4, 317), (242, 475)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'Finish')
            self.assertEqual(event['info'], '3')

    def test_third_s3(self):
        rv = video_loop('./testing/finish/third/t2/t2.mov', self.detector, [(320, 638), (0, 237)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'Finish')
            self.assertEqual(event['info'], '3')

    def test_fourth_emulator(self):
        rv = video_loop('./testing/finish/fourth/t1/t1.mov', self.detector, [(4, 317), (242, 475)], self.detector.detector_states)
        self.assertEqual(len(rv['events']), 1)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Lap')
            self.assertEqual(event['event_subtype'], 'Finish')
            self.assertEqual(event['info'], '4')


if __name__ == '__main__':
    unittest.main()
