#!/usr/bin/env python
"""Job dispatcher for MK64 analytics engine"""
import sys
# Fix pythonpath
sys.path.append('/usr/local/lib/python2.7/site-packages')

import argparse
import json
import multiprocessing
import os, pwd, grp, signal, syslog
import time
from subprocess import call

import api
import audiodetect as aud
import phase_0
import phase_1

from const import *

DAEMON = False
KILL_COUNT = 360



sqs = api.SQS(QUEUES)
s3 = api.S3(BUCKETS)
db = api.DB()

def log(message, level=syslog.LOG_INFO):
    if DAEMON:
        syslog.syslog(level, message)
    else:
        print message

def split_session(session_id, url, rv_queue, job_data):
    video_file = s3.download_url('session', url, session_id)
    if video_file is not None:
        # Process
        rv = phase_0.main(int(session_id), open(video_file))
        # Split session, update DB
        log("[Dispatcher] Phase 0 completed. Updating database...")
        parse_races('session', video_file, rv)

        db.update_session(session_id)
        # Cleanup
        os.remove(video_file)
        rv_queue.put(rv)
    else:
        # No video
        rv_queue.put(None)


def process_race(race_id, url, rv_queue, job_data):
    if job_data['processed'] is not True:
        video_file = s3.download_url('race', url, race_id)
        if video_file is not None:
            # Get player regions
            rv = phase_1.main(job_data['player_regions'], open(video_file), course=job_data['course'])
            log("[Dispatcher] Phase 1 finished. Updating database...")
            # Send events to database
            for race_vars in rv:
                db.post_events(race_id, race_vars['events'])
                payload = {"processed" : True}
                db.update_race(race_id, payload)
            # Cleanup
            os.remove(video_file)
            rv_queue.put(1)
        else:
            rv_queue.put(None)
    else:
        rv_queue.put(1)

def process_audio(race_id, url, rv_queue, job_data):
    audio_file = s3.download_url('audio', url, race_id)
    if audio_file is not None:
        rv = aud.detect(audio_file)
        db.post_events(race_id, rv)
        rv_queue.put(1)
    else:
        rv_queue.put(None)

def split_video(src, dst, start, duration):
    log("Splitting %s into %s" % (src, dst))
    command = ['ffmpeg', '-ss', str(start),'-i', src, '-y',
              '-vcodec', 'copy', '-acodec', 'copy',
               '-t', str(duration), dst ]
    ret = call(command)
    if ret != 0:
        # Command failed
        raise

def split_audio(src, dst, start, duration):
    command = ['ffmpeg', '-i', src, '-y',
                '-vn', '-ac', '2', '-ar', '16000', '-f', 'wav',
                '-ss', str(start), '-t', str(duration), dst]
    ret = call(command)
    if ret != 0:
        # Command failed
        raise

def cleanup():
    for worker in JOBS:
        worker.join()

def parse_races(event_type, video_file, rv):
    if event_type is 'session':
        for idx, race in enumerate(rv[0]['events']):
            ext = video_file.split('.')[-1]
            filename ='race%i.%s' % (idx, ext)
            wavfile = filename.split('.')[0]+'.wav'
            start = race['start_time']
            duration = race['duration']
            split_video(video_file, filename, start, duration)
            split_audio(video_file, wavfile, start, duration)

            # S3 Upload
            video_key = s3.upload(bucket='race-videos', file_path=filename)
            audio_key = s3.upload(bucket='race-audio', file_path=wavfile)

            # Clean
            os.remove(filename)
            os.remove(wavfile)

            # DB Notify
            vid_url = '%s%s' % (RACE_BUCKET_BASE, video_key)
            aud_url = '%s%s' % (AUDIO_BUCKET_BASE, audio_key)
            session_id = rv[0]['session_id']
            db_payload = {
                'video_url' : vid_url,
                'audio_url' : aud_url,
                'start_time' : race['start_time'],
                'duration' : race['duration'],
                'characters' : rv[0]['characters'],
                'course' : race['course'],
                'player_regions' : rv[0]['locked_regions'],
                'processed' : False,
                'video_split' : True
            }
            race_id = db.post_race(session_id, db_payload)
            # Notify audio queue
            audio_payload = {
                'id' : race_id,
                'video_url' : aud_url
            }
            sqs.write('audio-queue', json.dumps(audio_payload))

def daemonize():
    if os.fork() == 0:
        # Child
        os.setsid()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        pid = os.fork()
        if pid != 0:
            os._exit(0)
        else:
            return
    else:
        os._exit(0)

def drop_privileges(username='nobody'):
    if os.getuid() != 0:
        return
    uid = pwd.getpwnam(username).pw_uid
    gid = pwd.getpwnam(username).pw_gid
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    os.umask(077)

JOBS = list()
JOB_MAP = {
            'split-queue' : globals()['split_session'],
            'process-queue' : globals()['process_race'],
            'audio-queue' : globals()['process_audio']
          }


def dispatch_jobs():
    count = 0
    rv_queue = multiprocessing.Queue()
    try:
        while True:
            for q in QUEUES:
                # Check for phase0 jobs
                log('Listening on '+q)
                job = sqs.listen(q)
                if job is not None:
                    count = 0
                    log('Launching worker for ' + q)
                    job_data = json.loads(job['msg'].get_body())
                    worker = multiprocessing.Process(target=JOB_MAP[q],
                                                     args=(job['id'], job['url'],
                                                           rv_queue, job_data))
                    worker.start()
                    JOBS.append(worker)
                    # Wait for job to complete
                    worker.join()
                    rv = rv_queue.get()
                    if rv is not None:
                        log("Successfully completed job. Deleting from queue...")
                        sqs.delete_message(job['msg'])
            time.sleep(WAIT)
            count += 1
            if count >= KILL_COUNT:
                api.EC2.killself()
    except Exception as ex:
        log('Exiting due to: \n' + ex.message)
        cleanup()

def main():
    parser = argparse.ArgumentParser(description='Listens on SQS queues and dispatches jobs')
    parser.add_argument('-d', '--daemon', help="Runs the dispatcher as a daemon", action="store_true")
    parser.add_argument('--user', help="Specifies which user to drop daemon privileges to (Requires --daemon)", default='nobody')
    args = parser.parse_args()
    global DAEMON
    DAEMON = args.daemon
    if DAEMON:
        daemonize()
        drop_privileges(username=args.user)
        # Ensure we're in correct directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dispatch_jobs()

if __name__ == '__main__':
    main()
