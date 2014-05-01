#!/usr/bin/env python
"""Job dispatcher for MK64 analytics engine"""
import argparse
import json
import multiprocessing
import os, sys, pwd, grp, signal, syslog
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
rv_queue = multiprocessing.Queue()

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
        parse_events('session', video_file, rv)
        # Cleanup
        os.remove(video_file)
        rv_queue.put(rv)
    else:
        # No video
        rv_queue.put(None)


def process_race(race_id, url, rv_queue, job_data):
    video_file = s3.download_url('race', url, race_id)
    if video_file is not None:
        # Get player regions
        rv = phase_1.main(job_data['player_regions'], open(video_file))
        # Send events to database
        for race_vars in rv:
            db.post_events(race_id, race_vars['events'])
        # Cleanup
        rv_queue.put(1)
    else:
        rv_queue.put(None)

def process_audio(race_id, url, rv_queue):
    audio_file = s3.download_url('audio', url, race_id)
    if audio_file is not None:
        rv = aud.detect(audio)
        db.post_events(rv)
        rv_queue.put(1)
    else:
        rv_queue.put(None)

def split_video(src, dst, start, duration):
    log("Splitting %s into %s" % (src, dst))
    command = ['ffmpeg', '-i', src, '-y',
              '-vcodec', 'copy', '-acodec', 'copy',
              '-ss', str(start), '-t', str(duration), dst ]
    ret = call(command)
    if ret != 0:
        # Command failed
        raise

def split_audio(src, dst, start, duration):
    wav_name = dst.split('.')[0] + '.wav'
    command = ['ffmpeg', '-i', src, '-y',
                '-vn', '-ac', '2', '-ar', '44100', '-f', 'wav',
                '-ss', str(start), '-t', str(duration), wav_name]
    ret = call(command)
    if ret != 0:
        # Command failed
        raise

def cleanup():
    for worker in JOBS:
        worker.join()

def parse_events(event_type, video_file, rv):
    if event_type is 'session':
        for idx, race in enumerate(rv[0]['events']):
            ext = video_file.split('.')[-1]
            filename='race%i.%s' % (idx, ext)
            start = race['start_time']
            duration = race['duration']
            split_video(video_file, filename, start, duration)
            split_audio(video_file, filename, start, duration)

            # S3 Upload
            video_key = s3.upload(bucket='race-videos', file_path=filename)
            audio_key = s3.upload(bucket='race-audio', file_path=filename.split('.')[0]+'.wav')

            # Notify audio queue

            # Clean
            os.remove(filename)

            # DB Notify
            url = '%s%s' % (RACE_BUCKET_BASE, video_key)
            session_id = rv[0]['session_id']
            payload = {
                'video_url' : url,
                'start_time' : race['start_time'],
                'duration' : race['duration'],
                'characters' : rv[0]['characters'],
                'course' : rv[0]['map'],
                'player_regions' : rv[0]['locked_regions'],
                'processed' : False,
                'video_split' : True
            }
            db.post_race(session_id, payload)
    elif event_type is 'race':
        # Update DB with events dump
        pass
    elif event_type is 'audio':
        # Update DB with events dump
        pass


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
    try:
        while True:
            for q in QUEUES:
                # Check for phase0 jobs
                job = sqs.listen(q)
                log('Listening on '+q)
                if job is not None:
                    count = 0
                    log('Launching worker')
                    job_data = json.loads(job['msg'].get_body())
                    worker = multiprocessing.Process(target=JOB_MAP[q],
                                                     args=(job['id'], job['url'], rv_queue, job_data))
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
        print ex
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
    dispatch_jobs()

if __name__ == '__main__':
    main()
