import csv

"""DEBUG_LEVEL :Describes intensity of feedback from video processing
    = 0     No feedback
    = 1     Minor feedback      Displaying current frame, print object detections, etc
    = 2     Verbose feedback    Output intermediate values for more severe debugging
    = 3     More verbose        This level will most likely just be used in development
                                of features with unknown results
"""
DEBUG_LEVEL = 1

class Race(object):
    """Race state variables

    This class is for phase_0 use only. There will be 
    a separate class for phase_1. The idea behind this 
    separation is that both phases will have a different
    race state variable.
    """
    def __init__(self):
        """Sets the race's initial state"""
        self.race = {'session_id':    0,
                     'num_players':   0,
                     'p1':      "",
                     'p2':      "",
                     'p3':      "",
                     'p4':      "",
                     'start_time':    0,
                     'duration': 0,
                     'frame_rate':    0,}
        self.is_started = False
        self.player_boxes = list()
        self.is_black = False

    def reset(self):
        """ Resets race state dictionary (NOT the rest!)"""
        self.race.is_started = False
        self.race['start_time'] = 0
        self.race['duration'] = 0
        self.race['num_players'] = 0
        self.race['p1'] = ""
        self.race['p2'] = ""
        self.race['p3'] = ""
        self.race['p4'] = ""

    def save(filename):
        """Saves dictionary to specified file

        Should be used in "emergencies" due to DB failure
        """
        f = open(filename, "wb")
        w = csv.writer(f)
        for key, val in self.race.items():
            w.writerow([key, val])
        f.close()

#TODO: Does this need to be included here?
detector_states = dict()