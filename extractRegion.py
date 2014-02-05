#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="src", help="Directory containing masks")
    parser.add_option("-o", "--output", dest="dst", help="Directory to place pixelated masks into")
    parser.add_option("-x", "--startx", dest="startx", help="Starting x position")
    parser.add_option("-y", "--starty", dest="starty", help="Starting y position")
    parser.add_option("-l", "--height", dest="height", help="Height of the image to extract")
    parser.add_option("-w", "--width", dest="width", help="Width of the image to extract")
    (options, args) = parser.parse_args()
    print options
    # Error check
    if not all((options.src, options.dst, options.startx, options.starty, options.height, options.width)):
        print 'Incorrect usage.'
        exit(-1)

    masks = [(cv.imread(options.src+name), name) for name in os.listdir(options.src)]
    for m in masks:
        new_img = m[0][int(options.starty):(int(options.starty)+int(options.height)),int(options.startx):(int(options.startx)+int(options.width)),:]
        cv.imwrite(options.dst+'/'+m[1], new_img)
