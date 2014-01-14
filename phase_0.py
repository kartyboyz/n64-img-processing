#!/usr/bin/env python
import sys
import detection

def main(session_id, video_file):
    # Initialization
    r = detection.Engine(video_file.name)
    start_race = detection.StartRaceDetector(ROI_list=[
                                            ((167, 52), (182, 102))],
                                            masks_path='./pxls/pxl_start/',
                                            freq=1,
                                            threshold=0.06)
    race_end = detection.EndRaceDetector(session_id)
    items = detection.ItemDetector(ROI_list=[
                                            ((45, 34), (97, 74))],
                                            masks_path='./pxls/pxl_items/',
                                            freq=1,
                                            threshold=0.16,
                                            buf_len=8)
    # Prepare engine
    r.add_detector([start_race, race_end, items])
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
