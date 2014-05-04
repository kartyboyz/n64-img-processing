"""
DEBUG_LEVEL: Describes intensity of feedback from video processing.
    = 0     No feedback
    = 1     Minor feedback      Display correctly identified events
    = 2     Verbose feedback    Log all correct events as well as detected objects, Worker info
    = 3     Visual              Log all previously mentioned data + show frame and Workers
    = 4     Very Visual         Log all previouly mentioned data plus binary thresholds

    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""
import multiprocessing
DEBUG_LEVEL = 1

race = multiprocessing.Manager().dict({
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
        })

class player(object):
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.d = self.manager.dict({
           'frame_rate' : float(30),
           'course' : None,
           'place' : -1,
           'lap' : 1,
           'is_started' : False,
           'events' : list()
           })

    def __getitem__(self, item):
        return self.d[item]

    def __setitem__(self, key, value):
        self.d[key] = value
