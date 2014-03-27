#!/usr/bin/env python
"""Phase 0 of MK64 Detection Suite

Initial phase of detection, designed to generate data regarding
starts/ends of races in a MK64 gameplay video & send it back to the DB

Required Detectors:
    BlackFrame
        Used for optimization (skipping frames) and
        to determine rage-quits/end of race
    BoxExtractor
        Generates "box" data for Workers
    Characters
        Evaluates the characters chosen for the given race
    StartRace
        Finds Lakitu who indicates the start of a race
    EndRace
        Wrapper around BlackFrame to determine the end of a race
    Map
        Evaluates which MK64 map is being played
"""

import sys
import detection


def main(session_id, video_file):
    """Configuration Variables/Data Setup"""
    VARIABLES = [detection.config.race]

    """Detector Setup"""
    BLACK = detection.BlackFrame(variables=VARIABLES)
    BOXES = detection.BoxExtractor(variables=VARIABLES)
    ITEMS = detection.Items(masks_dir='./high_res_masks/item_masks/',
                            freq=1,
                            threshold=0.16,
                            default_shape=[(237, 314, 3)],
                            variables=VARIABLES,
                            buf_len=8)
    #TODO Fix thresh. for CHARS
    CHARS = detection.Characters(masks_dir='./high_res_masks/char_masks/',
                                 freq=1,
                                 threshold=0.10,
                                 default_shape=[(333, 318, 3)],
                                 variables=VARIABLES,
                                 buf_len=8)
    START_RACE = detection.StartRace(masks_dir='./high_res_masks/start_masks/',
                                     freq=1,
                                     threshold=0.17,
                                     default_shape=[(237, 318, 3), (237, 318, 3), (237, 344, 3)],
                                     variables=VARIABLES)
    END_RACE = detection.EndRace(variables=VARIABLES,
                                 session_id=session_id)
    """Engine Setup"""
    ENGINE = detection.Engine(variables=VARIABLES,
                              video_source=video_file.name)
    ENGINE.setup_processes(num=1,
                           regions=[None])
    ENGINE.add_detectors([BLACK, BOXES, START_RACE, END_RACE])

    """Main"""
    rv = ENGINE.process()
    print VARIABLES


def instructions():
    print "Debugger's Instructions:"
    print "\t<ESC> exits program"
    print "\t<space> pauses/unpauses current frame\n\n"

if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print "Please specify a session ID and video file."
        print "Usage:"
        print "%s <session ID> <video source>" % (sys.argv[0])
        exit(-1)
    instructions()
    main(int(sys.argv[1]), open(sys.argv[2]))
