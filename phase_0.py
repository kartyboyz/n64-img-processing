#!/usr/bin/env python
import sys
import detection
import config

def main(session_id, video_file):
    # Initialization
    race_vars = config.Race()
    r = detection.Engine(src=video_file.name, race_vars=race_vars)
    num_players = detection.PlayerNum(race_vars=race_vars)
    start_race = detection.StartRaceDetector(
                                            masks_path='./high_res_masks/start_masks/',
                                            freq=1,
                                            threshold=0.16,
                                            race_vars=race_vars)
    race_end = detection.EndRaceDetector(session_id, race_vars)
    
    # Prepare engine
    r.add_detector([start_race])
    # Process
    r.process()

if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print 'Please specify a session ID and video file.'
        print 'Usage:'
        print sys.argv[0], ' session_id video_source'
        exit(-1)
    print 'Instructions:'
    print '\t<ESC> exits program'
    print '\t<space> pauses/unpauses current frame\n\n'
    print sys.argv
    main(int(sys.argv[1]), open(sys.argv[2]))
