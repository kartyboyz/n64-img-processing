#!/usr/bin/env python
import sys
import detection

def main(session_id, video_file):
    # Initialization
    r = detection.Engine(video_file.name)

    num_players = detection.PlayerNum()
    start_race = detection.StartRaceDetector(
                                            masks_path='./high_res_masks/start_masks/',
                                            freq=1,
                                            threshold=0.16)
    race_end = detection.EndRaceDetector(session_id)
    
    # Prepare engine
    r.add_detector([PlayerNum])
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
