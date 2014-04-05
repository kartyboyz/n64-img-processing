"""DEBUG_LEVEL :Describes intensity of feedback from video processing
    = 0     No feedback
    = 1     Minor feedback      Displaying current frame, print object detections, etc
    = 2     Verbose feedback    Output intermediate values for more severe debugging
    = 3     More verbose        This level will most likely just be used in development
                                of features with unknown results
"""
import multiprocessing
DEBUG_LEVEL = 1

race = multiprocessing.Manager().dict({
        'session_id' : 0,
        'frame_rate' : float(30),
        'num_players' : 0,
        'start_time' : 0,
        'duration' : 0,
        'characters' : list(),
        'is_started' : False,
        'is_black' : False,
        'map' : "",
        'player_regions' : list(),
        'events' : list()
        })

player = multiprocessing.Manager().dict({
           'frame_rate' : float(30),
           'place' : -1,
           'lap' : -1,
           'is_started' : False,
           'events' : list()
           })

