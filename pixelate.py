#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from optparse import OptionParser

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

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("-i", "--input", dest="src", help="Directory containing masks")
	parser.add_option("-o", "--output", dest="dst", help="Directory to place pixelated masks into")
	parser.add_option("-r", "--resolution", dest="res", help="Resolution to pixelate at")
	parser.add_option("-d", "--display", dest="display", help="Directory for display images (optional)")
	(options, args) = parser.parse_args()
	print options
	# Error check
	if not all((options.src, options.dst, options.res)):
		print 'Incorrect usage.'
		exit(-1)

	masks = [(cv.imread(options.src+name), name) for name in os.listdir(options.src)]
	print masks
	for m in masks:
		cv.imshow('m', m[0])
		cv.waitKey()
		averaged, display = pixelate(m[0], int(options.res))
		cv.imwrite(options.dst+'/'+m[1], averaged)
		if options.display:
			cv.imwrite(options.display+'/'+m[1], display)