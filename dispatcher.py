#!/usr/bin/env python2.7
"""Job dispatcher for MK64 analytics engine"""
import json
import multiprocessing
import os
import time
from subprocess import call

import api
import phase_0
import phase_1

from const import *

def split_session(session_id, url, rv_queue, job_data):
    print 'Splitting session'
    video_file = s3.download_url('session', url, session_id)
    print video_file
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
    print video_file
    if video_file is not None:
        # Get player regions
        rv = phase_1.main(job_data['player_regions'], open(video_file))
        # Send events to database
        for race_vars in rv:
            print race_vars['events']
            db.post_events(race_id, race_vars['events'])
        # Cleanup
        rv_queue.put(1)
    else:
        rv_queue.put(None)

def process_audio(race_id, url, rv_queue):
    pass

def split_video(src, dst, start, duration):
    print "Splitting %s into %s" % (src, dst)
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



JOBS = list()
JOB_MAP = {
            'split-queue' : globals()['split_session'],
            'process-queue' : globals()['process_race'],
            'audio-queue' : globals()['process_audio']
          }

def parse_events(event_type, video_file, rv):
    print 'parsing events'
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


if __name__ == '__main__':
    sqs = api.SQS(QUEUES)
    s3 = api.S3(BUCKETS)
    db = api.DB()
    rv_queue = multiprocessing.Queue()

    # Listen & dispatch jobs
    while True:
        try:
            for q in QUEUES:
                print q
                # Check for phase0 jobs
                job = sqs.listen(q)
                print job
                if job is not None:
                    print 'Launching worker'
                    job_data = json.loads(job['msg'].get_body())
                    print job_data
                    worker = multiprocessing.Process(target=JOB_MAP[q],
                                                     args=(job['id'], job['url'], rv_queue, job_data))
                    worker.start()
                    JOBS.append(worker)
                    # Wait for job to complete
                    worker.join()
                    rv = rv_queue.get()
                    if rv is not None:
                        print "Successfully completed job. Deleting from queue..."
                        sqs.delete_message(job['msg'])
            time.sleep(WAIT)
        except KeyboardInterrupt:
            cleanup()
            exit()
