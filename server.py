from flask import Flask
import json
import os
import threading
import multiprocessing
import urllib2

import phase_0
import database
import tempfile

# System defaults. Will be overrideable at a later time
port = 5001

app = Flask(__name__)
app.debug = True

def download_race(video_url):
    f = tempfile.NamedTemporaryFile()
    chunk_size = 4 * 1024 * 1024
    try:
        req = urllib2.urlopen(video_url)
        while True:
            chunk = req.read(chunk_size)
            if not chunk: break
            f.write(chunk)
        f.flush()
        return f
    except urllib2.HTTPError:
        print "Couldn't download ", video_url
        return None
    except urllib2.URLError:
        print "Couldn't download ", video_url
        return None

def find_races(session_id, video_url):
    video_file = download_race(video_url)
    # Download movie
    if video_file is None:
        return -1
    phase_0.main(session_id, video_file)

@app.route('/race_detection/<int:session_id>', methods=['POST'])
def rcv_session_id(session_id):
    # Fetch URL
    session = database.get_session(session_id)
    video_url = session.json()[u'video_url']
    # Pass control to processing thread
    phase_0_thread = multiprocessing.Process(target = find_races, args=(session_id, video_url))
    phase_0_thread.start()
    return 'Launched job for session ID %d\n' % session_id

if __name__ == '__main__':
    app.run(port=port)
