
class Race(object):
    '''
    This class is for phase_0 use only. There will be 
    a separate class for phase_1. The idea behind this 
    separation is that both phases will have a different
    race state variable.
    '''
    def __init__(self):
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
        