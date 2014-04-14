import unittest
import cv2 as cv
import numpy as np

import detection
""" Unit tests for all Item cases

TODO: A lot of these could be automated further instead of redundant copy pasta
"""

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

    def test_1(self):
        rv = video_loop('./testing/items/Banana/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Banana')

    def test_2(self):
        rv = video_loop('./testing/items/Banana/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Banana')

    def test_3(self):
        rv = video_loop('./testing/items/Banana/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Banana')

    def test_4(self):
        rv = video_loop('./testing/items/Banana/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Banana')


class BananaBunchTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/BananaBunch/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BananaBunch')

    def test_2(self):
        rv = video_loop('./testing/items/BananaBunch/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BananaBunch')

    def test_3(self):
        rv = video_loop('./testing/items/BananaBunch/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BananaBunch')

    def test_4(self):
        rv = video_loop('./testing/items/BananaBunch/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BananaBunch')

class BlueShellTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/BlueShell/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BlueShell')

    def test_2(self):
        rv = video_loop('./testing/items/BlueShell/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BlueShell')

    def test_3(self):
        rv = video_loop('./testing/items/BlueShell/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BlueShell')

    def test_4(self):
        rv = video_loop('./testing/items/BlueShell/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'BlueShell')


class BooTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/Boo/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_2(self):
        rv = video_loop('./testing/items/Boo/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_3(self):
        rv = video_loop('./testing/items/Boo/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_4(self):
        rv = video_loop('./testing/items/Boo/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_5(self):
        rv = video_loop('./testing/items/Boo/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_6(self):
        rv = video_loop('./testing/items/Boo/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_7(self):
        rv = video_loop('./testing/items/Boo/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_8(self):
        rv = video_loop('./testing/items/Boo/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

    def test_9(self):
        rv = video_loop('./testing/items/Boo/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 2)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Boo')

class FakeItemTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/FakeItem/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'FakeItem')

    def test_2(self):
        rv = video_loop('./testing/items/FakeItem/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'FakeItem')

    def test_3(self):
        rv = video_loop('./testing/items/FakeItem/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'FakeItem')

    def test_4(self):
        rv = video_loop('./testing/items/FakeItem/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'FakeItem')


class GoldenMushroomTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/GoldenMushroom/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'GoldenMushroom')

    def test_2(self):
        rv = video_loop('./testing/items/GoldenMushroom/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'GoldenMushroom')

    def test_3(self):
        rv = video_loop('./testing/items/GoldenMushroom/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'GoldenMushroom')

    def test_4(self):
        rv = video_loop('./testing/items/GoldenMushroom/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'GoldenMushroom')


class LightningTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/Lightning/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Lightning')

    def test_2(self):
        rv = video_loop('./testing/items/Lightning/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Lightning')

    def test_3(self):
        rv = video_loop('./testing/items/Lightning/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Lightning')

    def test_4(self):
        rv = video_loop('./testing/items/Lightning/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Lightning')

class SingleGreenShellTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/SingleGreenShell/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleGreenShell')

    def test_2(self):
        rv = video_loop('./testing/items/SingleGreenShell/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleGreenShell')

    def test_3(self):
        rv = video_loop('./testing/items/SingleGreenShell/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleGreenShell')

    def test_4(self):
        rv = video_loop('./testing/items/SingleGreenShell/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleGreenShell')


class SingleMushroomTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/SingleMushroom/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleMushroom')

    def test_2(self):
        rv = video_loop('./testing/items/SingleMushroom/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleMushroom')

    def test_3(self):
        rv = video_loop('./testing/items/SingleMushroom/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleMushroom')

    def test_4(self):
        rv = video_loop('./testing/items/SingleMushroom/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleMushroom')


class SingleRedShellTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/SingleRedShell/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleRedShell')

    def test_2(self):
        rv = video_loop('./testing/items/SingleRedShell/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleRedShell')

    def test_3(self):
        rv = video_loop('./testing/items/SingleRedShell/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleRedShell')

    def test_4(self):
        rv = video_loop('./testing/items/SingleRedShell/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'SingleRedShell')

class StarTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/Star/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Star')

    def test_2(self):
        rv = video_loop('./testing/items/Star/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Star')

    def test_3(self):
        rv = video_loop('./testing/items/Star/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Star')

    def test_4(self):
        rv = video_loop('./testing/items/Star/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'Star')


class TripleGreenShellTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/TripleGreenShell/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleGreenShell')

    def test_2(self):
        rv = video_loop('./testing/items/TripleGreenShell/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleGreenShell')

    def test_3(self):
        rv = video_loop('./testing/items/TripleGreenShell/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleGreenShell')

    def test_4(self):
        rv = video_loop('./testing/items/TripleGreenShell/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleGreenShell')



class TripleMushroomTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/TripleMushroom/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleMushroom')

    def test_2(self):
        rv = video_loop('./testing/items/TripleMushroom/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleMushroom')

    def test_3(self):
        rv = video_loop('./testing/items/TripleMushroom/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleMushroom')

    def test_4(self):
        rv = video_loop('./testing/items/TripleMushroom/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleMushroom')


class TripleRedShellTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/TripleRedShell/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleRedShell')

    def test_2(self):
        rv = video_loop('./testing/items/TripleRedShell/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleRedShell')

    def test_3(self):
        rv = video_loop('./testing/items/TripleRedShell/t3/t3.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleRedShell')

    def test_4(self):
        rv = video_loop('./testing/items/TripleRedShell/t4/t4.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 1)
        event = rv['events'][0]
        self.assertEqual(event['event_type'], 'Items')
        self.assertEqual(event['event_subtype'], 'ItemGet')
        self.assertEqual(event['info'], 'TripleRedShell')


class FailCaseTest(unittest.TestCase):
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

    def test_1(self):
        rv = video_loop('./testing/items/FailCase/t1/t1.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 0)

    def test_2(self):
        rv = video_loop('./testing/items/FailCase/t2/t2.mov', self.detector, [(4, 317), (0, 237)])
        self.assertEqual(len(rv['events']), 0)

if __name__ == '__main__':
    unittest.main()
