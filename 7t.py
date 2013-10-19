import cv2 as cv
import cv as oldcv
import numpy as np
import os, sys


class Race:
	def __init__(self, src):
		self.name = src
		self.capture = cv.VideoCapture(src)
		self.ret, self.cur_frame = self.capture.read()
		self.avg_frames = np.float32(self.cur_frame)
		self.frame_cnt = 0
		cv.namedWindow(src, 1)
		cv.namedWindow('template', 1)
		print 'Race initialization complete.'

	def pixelate(self, image, resolution):
		h, w, d = image.shape
		block_size = (h/resolution, w/resolution)
		rv = np.zeros((resolution,resolution, 3), np.uint8)
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
				cv.rectangle(display, (block*c, block*r), (block*c+block, block*r+block), [avg_b, avg_g, avg_r], -1)
		return rv, display

	def manhattan(self, a, b):
		height, width = a.shape[:2]
		dist = 0
		for r in xrange(height):
			for c in xrange(width):
				dist += abs(np.int16(a[r][c]) - np.int16(b[r][c]))
		return sum(dist)


	def process(self):
		x=1
		# Init
		ITEMS = os.listdir('./items/')
		print ITEMS
		while True:
			ret, self.cur_frame = self.capture.read()
			p1_itembox = self.cur_frame[34:74, 45:97]
			f_avg, f_disp = self.pixelate(p1_itembox, resolution=4)

			for mask in ITEMS:
				print mask
				t = cv.imread('./items/'+mask)
				t_avg, t_disp = self.pixelate(t, resolution=4)

				distance = self.manhattan(t_avg, f_avg)
				print distance
				if distance <= 1100:
					print str(self.frame_cnt) +' contains ' + mask
					cv.imshow('frame',f_disp,)
					cv.imshow('template',t_disp)
					c = cv.waitKey(0)
					if c == 27:
						exit(0)
				cv.rectangle(self.cur_frame, (45, 34), (97, 74), [255, 0, 0], 2)
				cv.imshow(self.name, self.cur_frame)
			self.frame_cnt += 1
			c = cv.waitKey(x) % 0x100
			if c == 27:
				break
			elif c == 32:
				x = 0
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