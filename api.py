#!/usr/bin/env python
"""API around Amazon's boto library for SQS and S3 access"""
import boto
import json
import multiprocessing
import os
import time

from const import *

import phase_0
import phase_1

WAIT = 5
JOBS = list()

class SQS(object):
    def __init__(self, queue_names):
        self.conn = boto.connect_sqs(AWS_ACCESS_KEY_ID,
                                     AWS_SECRET_ACCESS_KEY)
        self.queues = {q : self.conn.get_queue(q) for q in queue_names}

    def listen(self, queue_name):
        try:
            q = self.queues[queue_name]
            raw_msg = q.get_messages(wait_time_seconds=WAIT)[0]
            msg = json.loads(raw_msg.get_body())
            url = str(msg['video_url'])
            video_id = str(msg['id'])
            function = JOB_MAP[queue_name]
            rv = {'function': function, 'msg':raw_msg,'id':video_id,'url':url}
            print rv
            return rv
        except:
            # No messages
            return None


class S3(object):
    def __init__(self, bucket_names):
        self.conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
                                    AWS_SECRET_ACCESS_KEY)
        self.buckets = {b : self.conn.get_bucket(b) for b in bucket_names}

    def download_url(self, video_type, video_url, video_id):
        filename = None
        if 'race' in video_type:
            name = 'race-videos'
        elif 'session' in video_type:
            name = 'session-videos'
        else:
            raise ValueError("Invalid video type")
        try:
            key_name = video_url.split('.com/')[-1]
            bucket = self.buckets[name]
            key = bucket.get_key(key_name)
            ext = video_url.split('.')[-1]
            filename = '%s%s.%s' % (name, video_id, ext)
            key.get_contents_to_filename(filename)
        except ValueError as e:
            print "Please specify either 'race-*' or 'session-*' as the type"
        except:
            print "Could not download the video"
        finally:
            return filename

def cleanup():
    for worker in JOBS:
        worker.join()

def split_session(session_id, url, rv_queue):
    print "%s: Downloading #%s from %s" % ('split-session', session_id, url)
    video_file = s3.download_url('session', url, session_id)
    if video_file is not None:
        # Process
        print video_file
        rv = phase_0.main(int(session_id), open(video_file))
        rv_queue.put(rv)
    else:
        # No video
        rv_queue.put(None)

def process_race(race_id, url):
    pass

JOB_MAP = {'split-queue' : globals()["split_session"], 'process-queue' : globals()["process_race"]}

if __name__ == '__main__':
    sqs = SQS(QUEUES)
    s3 = S3(BUCKETS)
    rv_queue = multiprocessing.Queue()

    # Listen & dispatch jobs
    while True:
        try:
            for q in QUEUES:
                # Check for phase0 jobs
                print q
                job = sqs.listen(q)
                if job is not None:
                    worker = multiprocessing.Process(target=job['function'],
                                                     args=(job['id'], job['url'], rv_queue))
                    worker.start()
                    JOBS.append(worker)
                    rv = rv_queue.get()
                    worker.join()
                    #if rv is not None:
                    #    sqs.delete_message(job['msg'])
            time.sleep(WAIT)
        except KeyboardInterrupt:
            cleanup()
            exit()
