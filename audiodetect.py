from detection import audioprocessing as proc

def detect(filename):
    audio_proc = proc.AudioAPI()
    fs,data = audio_proc.getwavdata(filename)
    if(len(data.shape)==2):
        track = audio_proc.whichone(data)
        sig = data[:,track]
    else:
        sig = data
    events = audio_proc.process(sig,fs)
    return events
