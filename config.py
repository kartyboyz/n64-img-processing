import json

class Race(object):
    def __init__(self):
        self.race = {'num_players':   0,
            'session_id':    0,
            'start_time':    0,
            'race_duration': 0,
            'frame_rate':    0}
        self.isStarted = False
