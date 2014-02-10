from collections import deque
import numpy as np
import cv2 as cv


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

def scaleImage(frame, mask, frame_shape_default):
    '''
    Generates a new image mask with dimensions scaled to the size of the video frame.
    This works by calculating the percent into the frame both (x1,y1) and (x2,y2) occur.
    (x1,y1) is the top left corner of the mask and (x2,y2) is the bottom right corner of the mask.
    Then compute the new x y pairs based on the calculated percentages and pass results onto cv.resize.
    Resizing operation is done via bilinear interpolation.
    '''

    # Get the dimensions of the frame and the shape of the mask
    h_frame, w_frame, _ = frame.shape
    h_mask, w_mask, _ = mask.shape
    coords = ((0,0),(w_mask,h_mask))
    
    x1_percentage = 0.0
    y1_percentage = 0.0
    x2_percentage = coords[1][0] / float(frame_shape_default[1]) * 100.0
    y2_percentage = coords[1][1] / float(frame_shape_default[0]) * 100.0
    
    newx1 = 0
    newy1 = 0
    newx2 = int(np.ceil((x2_percentage * float(w_frame)) / 100.0))
    newy2 = int(np.ceil((y2_percentage * float(h_frame)) / 100.0))

    # Resize the image
    fx = float(newx2) / float(w_mask)
    fy = float(newy2) / float(h_mask)
    scaled_image = cv.resize(mask, (newx2,newy2), fx, fy, cv.INTER_LINEAR)
    return scaled_image

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

    def exists(self, item):
        if self.count(item) > 0:
            return True
        else:
            return False

    def tolist(self):
        return list(self)
