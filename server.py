from flask import Flask
import json
import os
import threading
import urllib2
import requests

import phase_0
import database
import tempfile

# System defaults. Will be overrideable at a later time
port = 5001

app = Flask(__name__)
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

def find_races(session_id, video_file):
    phase_0.main(session_id, video_file)
    requests.post('http://localhost:5002/video_done/%d'% session_id)

@app.route('/race_detection/<int:session_id>', methods=['POST'])
def rcv_session_id(session_id):
    # Fetch URL
    session = database.get_session(session_id)
    video_url = session.json()[u'video_url']
    # Download movie
    video_file = download_race(video_url)
    if video_file is None:
        return -1
    # Pass control to processing thread
    # XXX: Limit total threads?
    #find_races(session_id, video_file)
    phase_0_thread = threading.Thread(target = find_races, args=(session_id, video_file))
    phase_0_thread.start()
    return 'Launched job for session ID %d\n' % session_id

if __name__ == '__main__':
    app.run(port=port)
