#!/usr/bin/env python
"""Phase 0 of MK64 Detection Suite

Initial phase of detection, designed to generate data regarding
starts/ends of races in a MK64 gameplay video & send it back to the DB

"""

import sys

import detection

def main(player_regions, video_file):
    """Configuration Variables/Data Setup"""
    VARIABLES = [detection.config.player for _ in range(len(player_regions))]
    """Detector Setup"""
    BLACK = detection.BlackFrame(variables=VARIABLES)
    ITEMS = detection.Items(masks_dir='./masks/items/',
                            freq=1,
                            threshold=0.16,
                            default_shape=[(237, 314, 3)],
                            variables=VARIABLES,
                            buf_len=8)
    BEGIN_RACE = detection.BeginRace(masks_dir='./masks/start/',
                                     freq=1,
                                     threshold=0.16,
                                     default_shape=[(237, 318, 3), (237, 344, 3)],
                                     variables=VARIABLES)
    FINISH_RACE = detection.FinishRace(masks_dir='./masks/finish/',
                                    freq=1,
                                    threshold=0.16,
                                    default_shape=[(237,314,3)],
                                    variables=VARIABLES)
    POSITION_CHANGE = detection.PositionChange(masks_dir='./masks/position/',
                                            freq=1,
                                            threshold=0.16,
                                            default_shape=[(237,314,3)],
                                            variables=VARIABLES,
                                            buf_len=2)
    SHORTCUT = detection.Shortcut(variables=VARIABLES)

    ENGINE = detection.Engine(variables=VARIABLES,
                              video_source=video_file.name)
    ENGINE.setup_processes(num=len(player_regions), regions=player_regions)
    ENGINE.add_detectors([BLACK])

    """Main"""
    rv = ENGINE.process()
    return rv


def instructions():
    print "Debugger's Instructions:"
    print "\t<ESC> exits program"
    print "\t<space> pauses/unpauses current frame\n\n"

if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print "Please specify a race ID and video file."
        print "Usage:"
        print "%s <race ID> <video source>" % (sys.argv[0])
        exit(-1)
    instructions()
    main(int(sys.argv[1]), open(sys.argv[2]))
