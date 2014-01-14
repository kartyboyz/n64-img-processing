import numpy as np
import cv2 as cv

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
