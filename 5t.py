import cv2 as cv
import numpy as np
import os, sys


'''
	This class contains all the necessary information for
	processing a video.
		name: Video source
		capture: Reference to video capture
		ret, cur_frame: Reference to frame being processed
		avg_frames: Stores moving average
		frame_cnt: Stores total # frames processed
		avg_weight: Decides 'retention rate' for frames
					in moving average. The higher the weight,
					the less the algorithm notices changes.
'''
class Race:
	def __init__(self, src):
		self.name = src
		self.capture = cv.VideoCapture(src)
		self.ret, self.cur_frame = self.capture.read()
		self.avg_frames = np.float32(self.cur_frame)
		self.frame_cnt = 0
		self.avg_weight = 0.2
		self.ROI = list()
		cv.namedWindow(src, 1)
		print 'Race initialization complete.\nBeginning processing...'

	def check_ROIs(self, p1, p2):
		for entry in self.ROI:
			if p1[0] >= 30 and p1[1] >= 20 and p2[0] <= 100 and p2[1] <= 80:
				print 'yeh'

	def process(self):
		self.ROI.append([((30,20), (100,80)), "item_p1"])


		while True:
			ret, self.cur_frame = self.capture.read()
			# Smooth out noise
			self.cur_frame = cv.GaussianBlur(self.cur_frame, (3, 3), 0)
			# Add frame to moving average to determine changes
			cv.accumulateWeighted(self.cur_frame, self.avg_frames, self.avg_weight, None)
			# Convert scale
			tmp = cv.convertScaleAbs(self.avg_frames)
			# Calculate difference between moving average and current
			diff = cv.absdiff(self.cur_frame, tmp)
			# Convert to grayscale --> binary
			imgray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
			thr, bin = cv.threshold(imgray, 70, 255, cv.THRESH_BINARY)

			# Dilate and erode to get blob-like features
			bin = cv.dilate(bin, None, iterations=18)
			bin = cv.erode(bin, None, iterations=10)
			# Extract contours
			contour, hierarchy = cv.findContours(bin,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
			# Draw bounding rectangles based on contour
			for cnt in contour:
				bound_rect = cv.boundingRect(cnt)
				p1 = (bound_rect[0], bound_rect[1])
				p2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
				cv.rectangle(self.cur_frame, p1, p2, (0,255,0),1)
				self.check_ROIs(p1, p2)
			cv.imshow(self.name, self.cur_frame)
			cv.imshow('BW', bin)

			self.frame_cnt += 1
			c = cv.waitKey(1) % 0x100
			if c == 27:
				break
			elif c == 32:
				cv.waitKey(0)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'Please specify video file.'
		exit(-1)
	print 'Instructions:'
	print '\t<ESC> exits program'
	print '\t<space> pauses/unpauses current frame\n\n'
	r = Race(sys.argv[1])
	r.process()