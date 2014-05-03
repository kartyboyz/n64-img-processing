"""API around S3, SQS, and MK64 database"""
import boto
import json
import multiprocessing
import os
import requests
import time
import uuid
import syslog

from subprocess import call
from boto.sqs.message import Message
from boto.sqs.message import RawMessage
from boto.exception import *
import boto.utils

from const import *

class EC2(object):
    def killself(self):
        asg = boto.connect_autoscale()
        instanceID = boto.utils.get_instance_identity()['document']['instanceId']
        asg.terminate_instance(instanceID)

    def get_launch_time(self):
        #TODO This.
        pass


class SQS(object):
    def __init__(self, queue_names):
        self.conn = boto.connect_sqs()
        self.queues = {q : self.conn.get_queue(q) for q in queue_names}

    def delete_message(self, msg):
        return msg.delete()

    def listen(self, queue_name):
        try:
            q = self.queues[queue_name]
            raw_msg = q.get_messages(wait_time_seconds=WAIT)[0]
            msg = json.loads(raw_msg.get_body())
            url = str(msg['video_url'])
            video_id = str(msg['id'])
            rv = {'msg':raw_msg,'id':video_id,'url':url}
            return rv
        except UnicodeDecodeError:
            filename = None
        except Exception as ex:
            # No messages
            return None

    def write(self, queue_name, payload):
        try:
            msg = Message()
            msg.set_body(payload)
            return self.queues[queue_name].write(msg)
        except SQSError:
            return None

class S3(object):
    def __init__(self, bucket_names):
        self.conn = boto.connect_s3()
        self.buckets = {b : self.conn.get_bucket(b) for b in bucket_names}

    def upload(self, bucket, file_path):
        k = boto.s3.key.Key(self.buckets[bucket])
        name = file_path.split('/')[-1]
        k.key = 'raw/%s/%s' % (str(uuid.uuid1()), name)
        k.set_contents_from_filename(file_path)
        return k.key

    def download_url(self, data_type, url, data_id):
        filename = None
        if 'race' in data_type:
            name = 'race-videos'
        elif 'session' in data_type:
            name = 'session-videos'
        elif 'audio' in data_type:
            name = 'race-audio'
        else:
            raise ValueError("Invalid video type")
        try:
            key_name = url.split('.com/')[-1]
            bucket = self.buckets[name]
            key = bucket.get_key(key_name)
            ext = url.split('.')[-1]
            filename = '%s%s.%s' % (name, data_id, ext)
            key.get_contents_to_filename(filename)
        except:
            filename = None
        finally:
            return filename

class DB(object):
    def __init__(self):
        self.database = 'http://n64storageflask-env.elasticbeanstalk.com'
        self.port = 80

    def get_regions(self, race_id):
        url = '%s:%d/races/%d' % (self.database, self.port, race_id)
        res = requests.get(url)
        if res.ok:
            return res.json()['player_regions']
        else:
            print 'DB Error: ' + res.json()['message']
            return None

    def post_events(self, race_id, events):
        responses = []
        url = '%s:%d/races/%s/events' % (self.database, self.port, race_id)
        for e in events:
            payload = json.dumps(e)
            header = {'Content-Type': 'application/json'}
            res = requests.post(url, data=payload, headers=header)
            if res.ok:
                responses.append(res.json()['id'])
            else:
                responses.append(None)
        return responses

    def post_race(self, session_id, payload):
        """Sends race JSON object to database for storage"""
        url = '%s:%d/sessions/%d/races' % (self.database, self.port, session_id)
        headers = {'content-type': 'application/json'}
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        if res.ok:
            return res.json()['id']
        else:
            return None

    def update_race(self, race_id, payload):
        """Sends race JSON object to database for storage"""
        url = '%s:%d/races/%d' % (self.database, self.port, race_id)
        headers = {'content-type': 'application/json'}
        res = requests.put(url, data=json.dumps(payload), headers=headers)
        return res.ok

    def update_session(self, session_id):
        """Sends race JSON object to database for storage"""
        url = '%s:%d/sessions/%d' % (self.database, self.port, session_id)
        headers = {'content-type': 'application/json'}
        payload = {
            'video_split' : True
        }
        res = requests.put(url, data=json.dumps(payload), headers=headers)
        return res.ok
