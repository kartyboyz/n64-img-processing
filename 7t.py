import cv2 as cv
import cv as oldcv
import numpy as np
import os, sys

class Detector:
	def __init__(self, pxl_masks, ROI, freq):
		self.masks = []
		for mask in pxl_masks:
			print mask
			self.masks.append((cv.imread(mask), mask))
		self.ROI = ROI
		self.freq = freq

	def detect(self, frame, count):
		if count % self.freq is 0:
			self.process(frame, count)

	def process(self, frame, count):
		print '[detector]: processing...'
		y0 = self.ROI[0][0]
		y1 = self.ROI[1][0]
		x0 = self.ROI[0][1]
		x1 = self.ROI[1][1]

		region = frame[x0:x1, y0:y1]
		f_avg, f_disp = self.pixelate(region, resolution=4)
		for mask in self.masks:
			t_avg, t_disp = self.pixelate(mask[0], resolution=4)
			distance = sum(sum(sum(abs(np.int16(f_avg) - np.int16(t_avg)))))
			cv.imshow('frame',f_disp,)
			cv.imshow('template',t_disp)
			if distance <= 1100:
				print str(count) +' contains ' + mask[1]
				c = cv.waitKey(0)
				if c == 27:
					exit(0)

	def pixelate(self, image, resolution):
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
				rv[r][c] = [avg_r, avg_g, avg_b]
				# Debug
				cv.rectangle(display, (block*c, block*r), (block*c+block, block*r+block), [avg_b, avg_g, avg_r], -1)
		return rv, display


class Race:
	def __init__(self, src):
		self.name = src
		self.capture = cv.VideoCapture(src)
		self.ret, self.cur_frame = self.capture.read()
		self.avg_frames = np.float32(self.cur_frame)
		self.frame_cnt = 0
		self.detectors = []
		# Debug
		cv.namedWindow(src, 1)
		print 'Race initialization complete.'

	def addDetectors(self, detectors):
		self.detectors.extend(detectors)
		print 'Added detector'

	def process(self):
		x=1
		# Init
		ITEMS = os.listdir('./items/')
		#print ITEMS
		while True:
			ret, self.cur_frame = self.capture.read()
			for d in self.detectors:
				d.detect(self.cur_frame, self.frame_cnt)
				cv.imshow(self.name, self.cur_frame)
			'''
			p1_itembox = self.cur_frame[34:74, 45:97]
			f_avg, f_disp = self.pixelate(p1_itembox, resolution=4)
			#print type(self.cur_frame)
			for mask in ITEMS:
				#print mask
				t = cv.imread('./items/'+mask)
				t_avg, t_disp = self.pixelate(t, resolution=4)
				distance = sum(sum(sum(abs(np.int16(f_avg) - np.int16(t_avg)))))
				#print distance
				if distance <= 1100:
					#print str(self.frame_cnt) +' contains ' + mask
					cv.imshow('frame',f_disp,)
					cv.imshow('template',t_disp)
					c = cv.waitKey(0)
					if c == 27:
						exit(0)
				cv.rectangle(self.cur_frame, (45, 34), (97, 74), [255, 0, 0], 2)
				cv.imshow(self.name, self.cur_frame)
			'''
			self.frame_cnt += 1
			c = cv.waitKey(x) % 0x100
			if c == 27:
				break
			elif c == 32:
				x ^=  1
				cv.waitKey(0)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'Please specify video file.'
		exit(-1)
	print 'Instructions:'
	print '\t<ESC> exits program'
	print '\t<space> pauses/unpauses current frame\n\n'
	r = Race(sys.argv[1])

	ITEMS = [os.path.join('./items/', f) for f in os.listdir('./items')]
	p1_itembox = Detector(ITEMS, [(45,34), (97, 74)], 1)
	p2_itembox = Detector(ITEMS, [(541,34), (593, 74)], 1)
	p3_itembox = Detector(ITEMS, [(45,264), (97, 304)], 1)
	p4_itembox = Detector(ITEMS, [(541, 264), (593, 304)], 1)
	r.addDetectors((p1_itembox, p2_itembox, p3_itembox, p4_itembox))
	r.process()