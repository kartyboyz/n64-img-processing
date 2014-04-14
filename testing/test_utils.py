def video_loop(video_src, detector, bounds, detector_states):
    """Iterates through video source, populating events"""

    cap = cv.VideoCapture(video_src)
    ret, frame = cap.read()
    count = 0
    while ret is not False:
        count += 1
        region = frame[bounds[1][0] : bounds[1][1],
                       bounds[0][0] : bounds[0][1]]
        if detector_states[detector.name()]:
            detector.detect(region, count, 0)
        ret, frame = cap.read()
    cap.release()
    return detector.variables
