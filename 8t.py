import cv2 as cv
import numpy as np
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
			rv[r][c] = [avg_r, avg_g, avg_b]
			# Debug
			cv.rectangle(display, (block*c, block*r), (block*c+block, block*r+block), [avg_b, avg_g, avg_r], -1)
	return rv, display

# Object to represent an averaged/downsized mask
class PixelMask:
	def __init__(self, path, name):
		self.name = name
		# Open image & store pixel matrix
		tmp = cv.imread(path)
		self.pxl_mask, self.pxl_display = pixelate(tmp, resolution=4)
		print '\t[PixelMask] added mask '+ self.name

class Detector:
	def __init__(self, ROI_list, masks_path, freq):
		self.ROI = ROI_list
		self.masks = [PixelMask(masks_path+name, name) for name in os.listdir(masks_path)]
		self.freq = freq
		print '[detector] initialization complete.'

	def detect(self, frame, cur_count):
		if cur_count % self.freq is 0:
			self.process(frame, cur_count)

	def process(self, frame, cur_count):
		for mask in self.masks:
			try:
				regions = iter(self.ROI)
			except:
				# Only one ROI to check
				y0 = self.ROI[0][0]
				y1 = self.ROI[1][0]
				x0 = self.ROI[0][1]
				x1 = self.ROI[1][1]
			else:
				# Multiple ROI to check
				for ROI in regions:
					y0 = ROI[0][0]
					y1 = ROI[1][0]
					x0 = ROI[0][1]
					x1 = ROI[1][1]

					# Pixelate current ROI in frame
					region = frame[x0:x1, y0:y1]
					f_pxl, f_disp = pixelate(region, resolution=4)

					# Compare for similarities
					distance = sum(sum(sum(abs(np.int16(f_pxl) - np.int16(mask.pxl_mask)))))

					if distance <= 1100:
						print str(cur_count) + ' contains ' + mask.name
						cv.rectangle(frame, (y0, x0), (y1, x1), (0, 255, 0))
						cv.imshow('compared to...', mask.pxl_display)
						c = cv.waitKey(0)
						if c == 27:
							exit(0)
					cv.imshow('itembox',f_disp)
					cv.imshow('frame',frame)

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
			ret, self.cur_frame = self.capture.read()
			for d in self.detectors:
				d.detect(self.cur_frame, self.frame_cnt)
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
	items_3p = Detector(ROI_list=[
							((45, 34), (97, 74)),
							((541,34), (593, 74)),
							((45, 264), (97, 304)),
							((541, 264), (593, 304))],
						masks_path='./items/',
						freq=1)
	r.add_detector(items_3p)
	r.process()
	cv.waitKey(0)
	cv.destroyAllWindows()