#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os, sys


# Constants
BLACK_FRAME_THRESHOLD = 6500000
BLACK_PXL_THRESHOLD = (50,50,50)
TRUE_BLACK = (0,0,0)

# Global function to average and downsize an image
def pixelate(image, resolution):
	h, w, _ = image.shape
	block_size = (h/resolution, w/resolution)
	rv = np.zeros((resolution, resolution, 3), np.uint8)
	# Debug
	display = np.zeros((160, 160, 3), np.uint8)
	block = 160/resolution
	for r in xrange(resolution):
		for c in xrange(resolution):
			# Determine coordinates
			start_x = block_size[1]*c
			start_y = block_size[0]*r
			# Calculate average
			avg_b, avg_g, avg_r, _ = cv.mean(image[start_y:start_y+block_size[0], start_x:start_x+block_size[1]])
			# Populate return matrix
			rv[r][c] = [avg_b, avg_g, avg_r]
			# Debug
			cv.rectangle(display, (block*c, block*r), (block*c+block, block*r+block), [avg_b, avg_g, avg_r], -1)
	return rv, display

# Ignores black pixels in 'mask', ONE CHANNEL ONLY
def manhattan(image, mask):
	# Mask out background noise
	image[mask <= BLACK_PXL_THRESHOLD] = mask[mask <= BLACK_PXL_THRESHOLD]
	# Manhattan distance
	manhattan = sum(sum(abs(np.int16(mask) - np.int16(image))))
	return manhattan

''' State Classes '''
class Player:
	def __init__(self):
		# Status indicators
		self.status = ''
		self.cur_item = None
		self.position = 0
		self.lap = 0

		# Trackers
		self.last_update = 0

		# Action-oriented variables
		self.in_collision = False
		self.in_shortcut = False
		self.in_boost = False
class Race:
	def __init__(self):
		self.num_players = 0
		self.start_frame = 0

# Global instantiation
race = Race()

# Parent class for all detectors
class Detector:
	def __init__(self, ROI_list, masks_path, freq, threshold_list):
		self.ROI_list = ROI_list
		self.masks = [(cv.imread(masks_path+name), name) for name in os.listdir(masks_path)]
		self.freq = freq
		self.threshold_list = threshold_list
		self.toggle = 0

	def detect(self, frame, cur_count):
		if cur_count % self.freq is 0:
			self.process(frame, cur_count)

	def process(self, frame, cur_count):
		# Player counter
		player = 1
		for ROI in self.ROI_list:
			# Extract ROI points
			col0 = ROI[0][0]
			col1 = ROI[1][0]
			row0 = ROI[0][1]
			row1 = ROI[1][1]
			# Pixelate current ROI in frame
			region = frame[row0:row1, col0:col1]
			f_pxl, f_disp = pixelate(region, resolution=4)
			for mask in self.masks:
				# Ignore black pixels in mask
				tmp_frame = f_pxl.copy()
				tmp_frame[(mask[0] <= BLACK_PXL_THRESHOLD).all(axis = -1)] = TRUE_BLACK
				mask[0][(mask[0] <= BLACK_PXL_THRESHOLD).all(axis = -1)] = TRUE_BLACK
				# Debug
				cv.imshow('FRAME', f_disp)
				cv.imshow('POST', tmp_frame)
				# Determine distances
				print mask[1]
				bf, gf, rf = cv.split(tmp_frame)
				bm, gm, rm = cv.split(mask[0])
				dif_b = sum(sum(abs(np.int16(bm) - np.int16(bf))))
				dif_g = sum(sum(abs(np.int16(gm) - np.int16(gf))))
				dif_r = sum(sum(abs(np.int16(rm) - np.int16(rf))))
				print dif_r, dif_g, dif_b
				tot = dif_b+dif_g+dif_r
				# Check values pass threshold test
				if all(map(lambda a,b: a <= b, [dif_b, dif_g, dif_r, tot], self.threshold_list)):
					# Transfer control to child class
					self.handle(frame, cur_count, player, mask)

				# DEBUG
				cv.imwrite('cur_f.png', tmp_frame)
				cv.imwrite('cur_m.png', mask[0])
				c = cv.waitKey(self.toggle)
				if c is 27:
					exit(0)
				elif c is 32:
					self.toggle ^= 1
					cv.waitKey(0)
			player += 1

class StartDetector(Detector):
	isStarted = False
	def handle(self, frame, cur_count, player, mask):
		x=0
		isStarted = True
		print '\t\tRace has started'
		c = cv.waitKey(x)
		if c is 27:
			exit(0)
		elif c is 32:
			x ^= 1
			cv.waitKey(0)

class FinishDetector(Detector):
	def handle(self, frame, cur_count, player, mask):
		x=0
		print 'Player ' + str(player) + ' gets ' + mask[1]
		c = cv.waitKey(x)
		if c is 27:
			exit(0)
		elif c is 32:
			x ^= 1
			cv.waitKey(0)

class PlayerNumDetector():
	def __init__(self):
		# Flags
		self.done = False
	def detect(self, frame, cur_count):
		if not self.done:
			# Threshold for true black
			# XXX/TODO: This is really slow for actual black frames :(
			frame[frame <= BLACK_PXL_THRESHOLD] = 0
			if np.sum(frame) > BLACK_FRAME_THRESHOLD:
				# Not black screen, check for lines
				self.process(frame, cur_count)

	def process(self, frame, cur_count):
		''' Modes:
				1 player:       No black lines intersecting video stream
				2 players:      One long horizontal black line in video stream
				3, 4 players:   Horizontal and vertical black lines in video stream
		'''
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		(h, w) = gray.shape
		# MK64-specific constants
		LINE_PX_H = h/2
		LINE_PX_W = w/2-3 # NOTE: -3 necessary because of asymmetry in frame
		# Threhsold for true black
		# Check ROI for black lines
		v_test = h_test = False
		black_count_w = np.sum(gray[:, LINE_PX_W] == 0)
		black_count_h = np.sum(gray[LINE_PX_H, :] == 0)
		if black_count_w >= h/2:
			h_test = True
		if black_count_h >= w/2:
			v_test = True

		if not v_test and h_test:
			print '2P mode'
		elif all((h_test, v_test)):
			print '3P or 4P mode'
		else:
			print '1p mode'


class ItemDetector(Detector):
	def handle(self, frame, cur_count, player, mask):
		print '\t\t\tPlayer ' + str(player) + ' has ' + mask[1]
		c = cv.waitKey(0)
		if c == 27:
			exit(0)


class Engine:
	def __init__(self, src):
		self.name = src
		self.capture = cv.VideoCapture(src)
		self.ret, self.cur_frame = self.capture.read()
		self.avg_frames = np.float32(self.cur_frame)
		self.frame_cnt = 1
		self.detectors = []
		# Debug
		cv.namedWindow(src, 1)
		print '[Engine] initialization complete.'

	def add_detector(self, detector):
		self.detectors.extend(detector)

	def process(self):
		x=1
		# Init
		while True:
			print '----[ Frame ' + str(self.frame_cnt) + ']----'
			ret, self.cur_frame = self.capture.read()
			cv.imshow(self.name, self.cur_frame)
			for d in self.detectors:
				d.detect(self.cur_frame, self.frame_cnt)

			self.frame_cnt += 1

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'Please specify video file.'
		exit(-1)
	print 'Instructions:'
	print '\t<ESC> exits program'
	print '\t<space> pauses/unpauses current frame\n\n'

# INITIALIZATION
	r = Engine(sys.argv[1])
	# Phase 0: Pre
	num_players = PlayerNumDetector()

	# Phase 1: Start
	# Phase 2: During
	items_3p = ItemDetector(ROI_list=[
							((45, 34), (97, 74)),
							((541,34), (593, 74)),
							((45, 264), (97, 304)),
							((541, 264), (593, 304))],
							masks_path='./pxl_items/',
							freq=1,
							threshold_list=[425, 425, 425, 1040])
	finish_race = FinishDetector(ROI_list=[
							((35, 112), (145, 222)),
							((495, 112), (605, 222)),
							((35, 332), (145, 422)),
							((495, 332), (605, 442))],
							masks_path='./pxl_finish/',
							freq=1,
							threshold_list=[500, 500, 500, 1400])

	# Prepare engine
	r.add_detector([finish_race])

# RUN
	r.process()
	cv.waitKey(0)
	cv.destroyAllWindows()