#!/usr/bin/env python
"""API around S3, SQS, and database"""
import boto
import json
import multiprocessing
import os
import requests
import time
import uuid

from subprocess import call
from boto.sqs.message import RawMessage

from const import *

class SQS(object):
    def __init__(self, queue_names):
        self.conn = boto.connect_sqs(AWS_ACCESS_KEY_ID,
                                     AWS_SECRET_ACCESS_KEY)
        self.queues = {q : self.conn.get_queue(q) for q in queue_names}

    def listen(self, queue_name):
        try:
            q = self.queues[queue_name]
            print 'listening on'
            print q
            raw_msg = RawMessage
            raw_msg = q.get_messages(wait_time_seconds=WAIT)[0]
            print raw_msg.get_body()
            msg = json.loads(raw_msg.get_body())
            print msg
            url = str(msg['video_url'])
            print url
            video_id = str(msg['id'])
            print video_id
            rv = {'msg':raw_msg,'id':video_id,'url':url}
            return rv
        except:
            # No messages
            return None

class S3(object):
    def __init__(self, bucket_names):
        self.conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
                                    AWS_SECRET_ACCESS_KEY)
        self.buckets = {b : self.conn.get_bucket(b) for b in bucket_names}

    def upload(self, bucket, file_path):
        k = boto.s3.key.Key(self.buckets[bucket])
        name = file_path.split('/')[-1]
        k.key = 'raw/%s/%s' % (str(uuid.uuid1()), name)
        k.set_contents_from_filename(file_path)
        return k.key

    def download_url(self, video_type, video_url, video_id):
        filename = None
        if 'race' in video_type:
            name = 'race-videos'
            key_name = video_url.split('.com/race-videos/')[-1]
        elif 'session' in video_type:
            name = 'session-videos'
            key_name = video_url.split('.com/session-videos/')[-1]
        else:
            raise ValueError("Invalid video type")
        try:
            print key_name
            bucket = self.buckets[name]
            print bucket
            key = bucket.get_key(key_name)
            print key
            ext = video_url.split('.')[-1]
            print ext
            filename = '%s%s.%s' % (name, video_id, ext)
            key.get_contents_to_filename(filename)
        except ValueError as e:
            print "Please specify either 'race-*' or 'session-*' as the type"
        finally:
            return filename

class DB(object):
    def __init__(self):
        self.server = 'http://n64storageflask-env.elasticbeanstalk.com'
        self.port = 80

    def get_session(self, session_id=0):
        """Communicates with database to extract information for a given session

        If session_id == 0, returns all sessions in database
        """
        session = '/sessions'
        if session_id is not 0:
            session += '/' + str(session_id)
        payload = '%s:%d%s' % (self.server, self.port, session)
        response = requests.get(payload)
        print response.text
        return response

    def get_races(self, session_id):
        """Communicates with database to extract information about races in session"""
        path = '/sessions/%d/races' % (session_id)
        payload = '%s:%d%s' % (self.server, self.port, path)
        response = requests.get(payload) # response.text contains the readable string
        return response

    def put_race(self,session_id, payload):
        """Sends race JSON object to database for storage"""
        url = '%s:%d/sessions/%d/races' % (self.server, self.port, session_id)
        headers = {'content-type': 'application/json'}
        print payload
        print type(payload['player_regions'][0][0][0])
        json_payload = json.dumps(payload)
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        return response
