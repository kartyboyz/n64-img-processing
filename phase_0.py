#!/usr/bin/env python

import sys
import detection
import config
import parallel

def main(session_id, video_file):
    VARIABLES = config.Race()
    DETECTOR_STATES = config.detector_states
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
    END_RACE = detection.EndRace(race_vars=VARIABLES, states=DETECTOR_STATES, session_id=session_id)

    ENGINE = detection.Engine(race_vars=VARIABLES, states=DETECTOR_STATES, video_source=video_file.name)
    ENGINE.setup_processes(num=1,
                           regions=[((0,0), (480, 640))])
                                    #[((0, 0), (237, 314)),
                                    #((239, 316), (476, 630))])
    ENGINE.add_detectors([START_RACE, END_RACE, CHARS, BOXES, BLACK])
    ENGINE.process()
    ENGINE.cleanup()

if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print "Please specify a session ID and video file."
        print "Usage:"
        print "%s <session ID> <video source>" % (sys.argv[0])
        exit(-1)
    print "Instructions:"
    print "\t<ESC> exits program"
    print "\t<space> pauses/unpauses current frame\n\n"
    print sys.argv
    main(int(sys.argv[1]), open(sys.argv[2]))
