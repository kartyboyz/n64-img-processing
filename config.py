
class Race(object):
    """Race state variables

    This class is for phase_0 use only. There will be 
    a separate class for phase_1. The idea behind this 
    separation is that both phases will have a different
    race state variable.
    """
    def __init__(self):
        """Sets the race's initial state"""
        self.race = {'num_players':   0,
            'p1':      0,
            'p2':      0,
            'p3':      0,
            'p4':      0,
            'session_id':    0,
            'start_time':    0,
            'race_duration': 0,
            'frame_rate':    0,}
        self.is_started = False
        self.player_boxes = list();
        self.is_black = False

    def reset(self):
        """ Resets race state dictionary (NOT the rest!)"""
        self.race.is_started = False
        self.race['start_time'] = 0
        self.race['race_duration'] = 0
        self.race['num_players'] = 0
        self.race['p1'] = 0
        self.race['p2'] = 0
        self.race['p3'] = 0
        self.race['p4'] = 0

#TODO: Does this need to be included here?
detector_states = dict()