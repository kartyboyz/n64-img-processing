#!/usr/bin/env python2.7
"""
Phase 1 of MK64 Detection Suite.
Processing phase of detection on each race.
    Authors: Johan Mickos   jmickos@bu.edu
             Josh Navon     navonj@bu.edu
"""

import sys
from detection.config import DEBUG_LEVEL

import detection
def debug_main(session_id, video_file, course):
    """Configuration Variables/Data Setup."""
    VARIABLES = [detection.config.player() for _ in range(4)]
    for player in VARIABLES:
        player['course'] = course

    # Detector Setup
    BLACK = detection.BlackFrame()
    ITEMS = detection.Items(masks_dir='./masks/items/',
                            freq=1,
                            threshold=0.16,
                            default_shape=[(237, 314, 3)],
                            buf_len=8)
    BEGIN_RACE = detection.BeginRace(masks_dir='./masks/start/',
                                     freq=1,
                                     threshold=0.16,
                                     default_shape=[(237, 314, 3), (237, 344, 3)])
    FINISH_RACE = detection.FinishRace(masks_dir='./masks/finish/',
                                    freq=1,
                                    threshold=0.16,
                                    default_shape=[(237,314,3)])
    POSITION_CHANGE = detection.PositionChange(masks_dir='./masks/position/',
                                            freq=1,
                                            threshold=0.16,
                                            default_shape=[(237,314,3)],
                                            buf_len=2)
    FALL = detection.Fall(masks_dir='./masks/fall/',
                        freq=1,
                        threshold=0.02,
                        default_shape=[(237, 305, 3)])
    REVERSE = detection.Reverse(masks_dir='./masks/reverse/',
                        freq=1,
                        threshold=0.05,
                        default_shape=[(237, 306, 3)])

    # [p1 test]: (11, 316), (2, 239)
    # [p1 s3]: (4, 317), (0, 237)
    # [p2 test]: (321, 629), (2, 241)
    # [p2 s3]: (320, 638), (0, 237)
    # [p3 test]: (11, 316), (244, 477)
    # [p3 s3]: (4, 317), (242, 475)
    # [p4 test]: (321, 629), (244, 477)
    # [p4 s3]: (320, 638), (242, 475)
    SHORTCUT = detection.Shortcut()
    LAP = detection.Lap(masks_dir='./masks/laps/',
                        freq=1,
                        threshold=0.08,
                        default_shape=[(237,314,3)])
    # Engine Setup
    ENGINE = detection.Engine(variables=VARIABLES,
                              video_source=video_file.name)
    ENGINE.setup_processes(num=4, regions=[[[4, 317], [0, 237]], [[320, 638], [0, 237]], [[4, 317], [242, 475]], [[320, 638], [242, 475]]])

    ENGINE.add_detectors([BLACK, LAP, BEGIN_RACE, FINISH_RACE, POSITION_CHANGE, SHORTCUT, ITEMS, FALL, REVERSE])

    # Main
    rv = ENGINE.process()
    return rv


def main(player_regions, video_file, course):
    """Configuration Variables/Data Setup."""
    VARIABLES = [detection.config.player() for _ in range(len(player_regions))]
    for player in VARIABLES:
        player['course'] = course

    # Detector Setup
    BLACK = detection.BlackFrame()
    ITEMS = detection.Items(masks_dir='./masks/items/',
                            freq=1,
                            threshold=0.16,
                            default_shape=[(237, 314, 3)],
                            buf_len=8)
    BEGIN_RACE = detection.BeginRace(masks_dir='./masks/start/',
                                     freq=1,
                                     threshold=0.16,
                                     default_shape=[(237, 314, 3), (237, 344, 3)])
    FINISH_RACE = detection.FinishRace(masks_dir='./masks/finish/',
                                    freq=1,
                                    threshold=0.16,
                                    default_shape=[(237,314,3)])
    POSITION_CHANGE = detection.PositionChange(masks_dir='./masks/position/',
                                            freq=1,
                                            threshold=0.16,
                                            default_shape=[(237,314,3)],
                                            buf_len=2)
    SHORTCUT = detection.Shortcut()
    LAP = detection.Lap(masks_dir='./masks/laps/',
                        freq=1,
                        threshold=0.08,
                        default_shape=[(237,314,3)])

    # Engine Setup
    ENGINE = detection.Engine(variables=VARIABLES,
                              video_source=video_file.name)
    ENGINE.setup_processes(num=len(player_regions), regions=player_regions)
    ENGINE.add_detectors([BLACK, LAP, BEGIN_RACE, FINISH_RACE, POSITION_CHANGE, SHORTCUT, ITEMS])

    # Main
    rv = ENGINE.process()
    return VARIABLES


def instructions():
    """Outputs instructions to stdout."""
    print "Debugger's Instructions:"
    print "\t<ESC> exits program"
    print "\t<space> pauses/unpauses current frame\n\n"

if __name__ == '__main__':
    if len(sys.argv) is not 4:
        print "Please specify a race ID and video file."
        print "Usage:"
        print "%s <race ID> <video source> <course-name>" % (sys.argv[0])
        exit(-1)
    instructions()
    rv = debug_main(int(sys.argv[1]), open(sys.argv[2]), sys.argv[3])
    if rv != None:
        for ii in xrange(len(rv)):
            if len(rv[ii]['events']) != 0 and rv[ii]['events'][-1]['event_info'] == "KoopaTroopaBeachCave":
                temp = rv[ii]['events']
                temp.pop()
                rv[ii]['events'] = temp
            for event in rv[ii]['events']:
                print event
    if DEBUG_LEVEL > 1:
        with open("./testing/events.log", "a") as event_log:
            for event in rv[0]['events']:
                event_log.write(str(event) + "\n\n")
