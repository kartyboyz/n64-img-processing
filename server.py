import requests
import json
from flask import Flask

import phase_0
import database

# System defaults. Will be overrideable at a later time
server = 'http://localhost'
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
        print "Couldn't download ", session
        return None
    except urllib2.URLError:
        print "Couldn't download ", session
        return None

def find_races(session_id, video_file):
    phase_0.main(session_id, video_file)

@app.route('/race_detection/<int:session_id>')
def rcv_session_id(session_id):
    # Fetch URL
    session = database.get_session(session_id)
    video_url = session.json()[u'video_url']
    # Download movie
    video_file = download_race(video_url)
    # Pass control to processing
    find_races(session_id, video_file)
    return 'Completed phase 0 for session_id %d\n' % session_id

if __name__ == '__main__':
    app.run(port=port)
