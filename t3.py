import cv2 as cv
import numpy as np
import sys, os, math



'''
	Scan pixel-by-pixel, stop after N consecutive
'''

def convert(image):
	h, w, d = image.shape
	block_size = (h/8, w/8)
	resolution = 8
	rv = np.zeros((8,8, 3), np.uint8)
	display = np.zeros((160, 160, 3), np.uint8)

	for r in xrange(resolution):
		for c in xrange(resolution):
			# Determine coordinates
			start_x = block_size[1]*c
			start_y = block_size[0]*r
			# Calculate average
			avg_b, avg_g, avg_r, _ = cv.mean(image[start_y:start_y+block_size[0], start_x:start_x+block_size[1]])
			# Populate return matrix
			rv[r][c] = [avg_r, avg_g, avg_b]
			cv.rectangle(display, (20*c, 20*r), (20*c+20, 20*r+20), [avg_b, avg_g, avg_r], -1)
	return rv, display

# Main Tests
for frame in os.listdir("./frames"):
	for template in os.listdir("./items"):
		f = cv.imread("./frames/"+frame)
		t = cv.imread("./items/"+template)
		h, w, _ = f.shape
		if (w,h) is not (640,480):
			f = cv.resize(f, (640,480))
		f = f[39:94, 285:354]
		f_avg, f_disp = convert(f)
		t_avg, t_disp = convert(t)
		dif_r = dif_g = dif_b = 0
		print f_avg
		for r in xrange(8):
			for c in xrange(8):
				dif_r += max(f_avg[r][c][2], t_avg[r][c][2]) - min(f_avg[r][c][2], t_avg[r][c][2])
				dif_g += max(f_avg[r][c][1], t_avg[r][c][1]) - min(f_avg[r][c][1], t_avg[r][c][1])
				dif_b += max(f_avg[r][c][0], t_avg[r][c][0]) - min(f_avg[r][c][0], t_avg[r][c][0])
		print dif_b+dif_g+dif_r
		if (dif_b + dif_g + dif_r) < 5000:
			print("%s contains %s" %(frame, template))
			cv.imshow('frame',f_disp)
			cv.imshow('template',t_disp)
			cv.imwrite('new.jpg', f)
			cv.waitKey(0)
cv.destroyAllWindows()

