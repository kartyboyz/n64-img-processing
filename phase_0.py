#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os, sys
import detection


def test(session_id, video_file):
	print '[PHASE 0]: ', session_id, video_file

def main(session_id, video_file):
	# INITIALIZATION
	r = detection.Engine(video_file.name)
	# Phase 0: Pre
	num_players = detection.PlayerNumDetector()
	start_race = detection.StartDetector(ROI_list=[
			((167, 52), (182, 102))],
				   masks_path='./pxls/pxl_start/',
				   freq=1,
				   threshold_list=[425, 425, 425, 1040])
	race_end = detection.EndRaceDetector(session_id)
	# Prepare engine
	r.add_detector([start_race, race_end])

	# RUN
	r.process()
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'Please specify video file.'
		exit(-1)
	print 'Instructions:'
	print '\t<ESC> exits program'
	print '\t<space> pauses/unpauses current frame\n\n'
	print sys.argv
	main(int(sys.argv[1]), open(sys.argv[2]))
