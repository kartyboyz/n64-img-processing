import numpy as np
import cv2 as cv
import os, sys

# Universal function to average and downsize an image
def pixelate(image, resolution):
	h, w, d = image.shape
	block_size = (h/resolution, w/resolution)
	rv = np.zeros((resolution,resolution, 3), np.uint8)
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

# Parent class for all detectors
class Detector:
	def __init__(self, ROI_list, masks_path, freq):
		self.ROI_list = ROI_list
		self.masks = [(cv.imread(masks_path+name), name) for name in os.listdir(masks_path)]
		self.freq = freq

	def detect(self, frame, cur_count):
		if cur_count % self.freq is 0:
			self.process(frame, cur_count)

	def process(self, frame, cur_count):
		# Player counter
		player = 1
		# Flow control
		x = 1
		for ROI in self.ROI_list:
			# Extract ROI points
			y0 = ROI[0][0]
			y1 = ROI[1][0]
			x0 = ROI[0][1]
			x1 = ROI[1][1]
			# Pixelate current ROI in frame
			region = frame[x0:x1, y0:y1]
			f_pxl, f_disp = pixelate(region, resolution=4)
			bf, gf, rf = cv.split(f_pxl)
			#cv.imshow('frame', f_disp)
			for mask in self.masks:
				print mask[1]
				bm, gm, rm = cv.split(mask[0])
				dif_b = sum(sum(abs(np.int16(bm) - np.int16(bf))))
				dif_g = sum(sum(abs(np.int16(gm) - np.int16(gf))))
				dif_r = sum(sum(abs(np.int16(rm) - np.int16(rf))))
				print dif_r, dif_g, dif_b
				tot = dif_b+dif_g+dif_r
				if dif_b < 420 and dif_g < 420 and dif_r < 420 and tot <= 1020:
					# Transfer control to child class
					self.handle(frame, cur_count, player, mask)
			c = cv.waitKey(x)
			if c is 27:
				exit(0)
			elif c is 32:
				x ^= 1
				cv.waitKey(0)
			player += 1
''' SPECIFIC detectors
* ItemDetector: handles itemboxes for all players
'''
class ItemDetector(Detector):
	def handle(self, frame, cur_count, player, mask):
		print '\t\t\tPlayer ' + str(player) + ' has ' + mask[1]
		c = cv.waitKey(0)
		if c == 27:
			exit(0)

class Race:
	def __init__(self, src):
		self.name = src
		self.capture = cv.VideoCapture(src)
		self.ret, self.cur_frame = self.capture.read()
		self.avg_frames = np.float32(self.cur_frame)
		self.frame_cnt = 1
		self.detectors = []
		# Debug
		cv.namedWindow(src, 1)
		print '[Race] initialization complete.'

	def add_detector(self, detector):
		self.detectors.append(detector)

	def process(self):
		x=1
		# Init
		while True:
			print 'NEWFRAME------------'
			ret, self.cur_frame = self.capture.read()
			cv.imshow(self.name, self.cur_frame)
			cv.waitKey(1)
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

	r = Race(sys.argv[1])
	items_3p = ItemDetector(ROI_list=[
							((45, 34), (97, 74)),
							((541,34), (593, 74)),
							((45, 264), (97, 304)),
							((541, 264), (593, 304))],
						masks_path='./pxl_items/',
						freq=1)
	r.add_detector(items_3p)
	r.process()
	cv.waitKey(0)
	cv.destroyAllWindows()