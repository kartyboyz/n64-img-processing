from collections import deque
import numpy as np
import cv2 as cv

def enum(*sequetial, **named):
    enums = dict(zip
                    (sequential,
                     range(length(sequential))),
                    **named)
    return type('Enum', (), enums)

# Global function to average and downsize an image
def pixelate(image, resolution):
    '''
    Generates a new image with dimesions [resolution]x[resolution],
    with each cell containing the average color of its corresponding
    block in the 'image' parameter
    '''
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
            # Calculate average of current block
            avg_b, avg_g, avg_r, _ = cv.mean(image[start_y:start_y+block_size[0], start_x:start_x+block_size[1]])
            # Populate return matrix
            rv[r][c] = [avg_b, avg_g, avg_r]
            # Debug
            cv.rectangle(display, (block*c, block*r), (block*c+block, block*r+block), [avg_b, avg_g, avg_r], -1)
    return rv, display


class RingBuffer(deque):
    def __init__(self, max_size):
        deque.__init__(self)
        self.__max_size = max_size

    def append(self, x):
        deque.append(self, x)
        if len(self) == self.__max_size:
            self.popleft()

    def all_same(self):
        if self.count(self[0]) is (self.__max_size-1):
            return True;
        else:
            return False;

    def tolist(self):
        return list(self)