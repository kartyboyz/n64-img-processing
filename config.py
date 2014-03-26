import csv

"""DEBUG_LEVEL :Describes intensity of feedback from video processing
    = 0     No feedback
    = 1     Minor feedback      Displaying current frame, print object detections, etc
    = 2     Verbose feedback    Output intermediate values for more severe debugging
    = 3     More verbose        This level will most likely just be used in development
                                of features with unknown results
"""
DEBUG_LEVEL = 1

race = {
        'session_id' : 0,
        'frame_rate' : float(30),
        'num_players' : 0,
        'start_time' : 0,
        'duration' : 0,
        'p1' : "",
        'p2' : "",
        'p3' : "",
        'p4' : "",
        'is_started' : False,
        'is_black' : False,
        'player_boxes' : list()
        }

player = {
           'frame_rate' : -1,
           'place' : -1,
           'lap' : -1,
           'item' : None,
           'in_collision' : False,
           'in_shortcut' : False,
         }

