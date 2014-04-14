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
        for d in detector:
            if d.detector_states[d.name()]:
                if isinstance(d, detection.BlackFrame):
                    d.detect(frame, count)
                else:
                    d.detect(region, count, 0)
        ret, frame = cap.read()
    cap.release()
    return detector[0].variables


class MapTest(unittest.TestCase):
    def setUp(self):
        self.detector = [detection.BlackFrame(), 
                        detection.Map(masks_dir='./masks/maps/',
                                    freq=1,
                                    threshold=0.16,
                                    default_shape=[(475, 619, 3)])]
        self.states = dict()

        self.variables = {
            'session_id' : 0,
            'frame_rate' : float(30),
            'num_players' : 1,
            'characters' : list(),
            'is_started' : False,
            'is_black' : False,
            'map' : "",
            'player_regions' : list(),
            'events' : list(),
            'locked_regions' : list()
        }
        for d in self.detector:
            self.states[d.name()] = True
            d.set_variables(self.variables)
            d.set_states(self.states)

    def test_map_source(self):
        rv = video_loop('./testing/maps/BansheeBoardwalk/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'BansheeBoardwalk')

        rv = video_loop('./testing/maps/BowsersCastle/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'BowsersCastle')
        
        rv = video_loop('./testing/maps/ChocoMountain/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'ChocoMountain')
        
        rv = video_loop('./testing/maps/DKsJungleParkway/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'DKsJungleParkway')
        
        rv = video_loop('./testing/maps/FrappeSnowland/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'FrappeSnowland')
        
        rv = video_loop('./testing/maps/KalimariDesert/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'KalimariDesert')
        
        rv = video_loop('./testing/maps/KoopaTroopaBeach/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'KoopaTroopaBeach')
        
        rv = video_loop('./testing/maps/LuigiRaceway/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'LuigiRaceway')
        
        rv = video_loop('./testing/maps/MarioRaceway/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'MarioRaceway')
        
        rv = video_loop('./testing/maps/MooMooFarm/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'MooMooFarm')
        
        rv = video_loop('./testing/maps/RainbowRoad/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'RainbowRoad')
        
        rv = video_loop('./testing/maps/RoyalRaceway/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'RoyalRaceway')
        
        rv = video_loop('./testing/maps/SherbetLand/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'SherbetLand')
        
        rv = video_loop('./testing/maps/WarioStadium/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'WarioStadium')
        
        rv = video_loop('./testing/maps/YoshiValley/t1/t1.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], 'YoshiValley')

    def test_FailCase(self):
        rv = video_loop('./testing/maps/FailCase/FailCase.mov', self.detector, [(11, 631), (2, 477)])
        self.assertEqual(rv['map'], "")


if __name__ == '__main__':
    unittest.main()
