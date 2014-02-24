#!/usr/bin/env python
import sys
import detection
import config

def main(session_id, video_file):
    # Initialization
    race_vars = config.Race()
    detector_states = config.detector_states
    r = detection.Engine(src=video_file.name, race_vars=race_vars, detector_states=detector_states)
    box_extractor = detection.BoxExtractor(race_vars=race_vars)
    start_race = detection.StartRaceDetector(
                                            masks_path='./high_res_masks/start_masks/',
                                            freq=1,
                                            threshold=0.16,
                                            race_vars=race_vars,
                                            default_frame=(237, 314, 3))
    end_race = detection.EndRaceDetector(session_id, race_vars)
    black_frames = detection.BlackFrameDetector(race_vars)
    char_detector = detection.CharacterDetector(
                                                masks_path='./high_res_masks/char_masks/',
                                                freq=2,
                                                threshold=0.034,
                                                race_vars=race_vars,
                                                default_frame=(333, 318, 3),
                                                buf_len=8)

    # Prepare engine
    r.add_detector([black_frames, box_extractor, start_race, end_race, char_detector])
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
