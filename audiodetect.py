from detection import audioprocess as proc

import numpy as np

def detect_events(filename):
	fs,data = proc.getwavdata(filename)
	if(len(data.shape)==2):
		track = proc.whichone(data)
		sig = data[:,track]
	else:
		sig = data
	events = proc.process(sig,fs)

	return events