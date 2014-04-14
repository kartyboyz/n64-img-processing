import unittest
import cv2 as cv
import numpy as np

import detection

def video_loop(video_src, detector, bounds):
    """Iterates through video source, populating events"""
    cap = cv.VideoCapture(video_src)
    ret, frame = cap.read()
    count = 0
    while ret is not False:
        count += 1
        region = frame[bounds[1][0] : bounds[1][1],
                       bounds[0][0] : bounds[0][1]]
        detector.detect(region, count, 0)
        ret, frame = cap.read()
    return detector.variables

class BananaTest(unittest.TestCase):
    def setUp(self):
        self.detector = detection.Items(masks_dir='./masks/items/',
                            freq=1,
                            threshold=0.16,
                            default_shape=[(237, 314, 3)],
                            buf_len=8)
        self.variables = {
            'events' : list(),
            'lap' : 1,
            'place' : -1,
            'is_started' : False,
            'frame_rate' : float(30)
        }
        self.detector.set_variables(self.variables)

    def test_banana(self):
        rv = video_loop('./testing/items/Banana/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertNotEqual(len(rv['events']), 0)
        for event in rv['events']:
            self.assertEqual(event['event_type'], 'Items')
            self.assertEqual(event['event_subtype'], 'ItemGet')
            self.assertEqual(event['info'], 'Banana')

if __name__ == '__main__':
    unittest.main()
