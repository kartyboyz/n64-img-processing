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
"""

import sys

import detection
import config
import parallel

from config import DEBUG_LEVEL

def main(session_id, video_file):
    """Configuration Variables/Data Setup"""
    VARIABLES = config.Race()
    DETECTOR_STATES = config.detector_states

    """Detector Setup"""
    BLACK = detection.BlackFrame(race_vars=VARIABLES, states=DETECTOR_STATES,)
    BOXES = detection.BoxExtractor(race_vars=VARIABLES, states=DETECTOR_STATES,)
    ITEMS = detection.Items(masks_dir='./high_res_masks/item_masks/',
                            freq=1,
                            threshold=0.16,
                            default_shape=(237, 314, 3),
                            race_vars=VARIABLES,
                            states=DETECTOR_STATES,
                            buf_len=8)
    #TODO Fix thresh. for CHARS
    CHARS = detection.Characters(masks_dir='./high_res_masks/char_masks/',
                                 freq=1,
                                 threshold=0.10,
                                 default_shape=(333, 318, 3),
                                 race_vars=VARIABLES,
                                 states=DETECTOR_STATES,
                                 buf_len=8)
    START_RACE = detection.StartRace(masks_dir='./high_res_masks/start_masks/',
                                     freq=1,
                                     threshold=0.16,
                                     default_shape=(237, 314, 3),
                                     race_vars=VARIABLES,
                                     states=DETECTOR_STATES)
    END_RACE = detection.EndRace(race_vars=VARIABLES,
                                 states=DETECTOR_STATES,
                                 session_id=session_id)

    """Engine Setup"""
    ENGINE = detection.Engine(race_vars=VARIABLES,
                              states=DETECTOR_STATES,
                              video_source=video_file.name)
    ENGINE.setup_processes(num=1,
                           regions=[None])
    ENGINE.add_detectors([BLACK, BOXES, CHARS, START_RACE, END_RACE])

    """Main"""
    ENGINE.process()
    ENGINE.cleanup()

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
    if DEBUG_LEVEL is not 0:
        instructions()
    main(int(sys.argv[1]), open(sys.argv[2]))
