import requests
from flask import Flask
import json

# System defaults. Will be overrideable at a later time
server = 'http://localhost'
port = 5000

def get_session(session_id):
    '''
    Communicates with database to extract information for a given session.
    If session_id == 0, returns all sessions in database
    '''
    global server, port
    session = '/sessions'
    if session_id is not 0:
        session += '/' + str(session_id)
    payload = '%s:%d%s'%(server, port, session)
    response = requests.get(payload)
    print response.text
    return response

def get_races(session_id):
    '''
    Communicates with database to extract information about races in session
    '''
    global server, port
    path = '/sessions/%d/races'%(session_id)
    payload = '%s:%d%s'%(server, port, path)
    response = requests.get(payload)
    print response.text
    return response

def put_race(session_id, start_time, duration):
    '''
    Sends race JSON object to database for storage.
    '''
    global server, port
    url = '%s:%d/sessions/%d/races'%(server, port, session_id)
    payload = {'start_time' : start_time, 'duration' : duration}
    headers = {'content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print response.text
    return response

