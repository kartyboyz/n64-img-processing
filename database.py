import json
import requests

from config import DEBUG_LEVEL

# System defaults. Will be overrideable at a later time
server = 'http://localhost'
port = 5000

def get_session(session_id):
    """Communicates with database to extract information for a given session

    If session_id == 0, returns all sessions in database
    """
    global server, port
    session = '/sessions'
    if session_id is not 0:
        session += '/' + str(session_id)
    payload = '%s:%d%s' % (server, port, session)
    response = requests.get(payload)
    print response.text
    return response

def get_races(session_id):
    """Communicates with database to extract information about races in session"""
    global server, port
    path = '/sessions/%d/races' % (session_id)
    payload = '%s:%d%s' % (server, port, path)
    response = requests.get(payload) # respond.text contains the readable string
    return response

def get_playerROI(session_id):
    """Communicates with DB to extract player boxes from the session"""
    global server, port
    # TODO: Figure out payload, parsing, return value

def put_race(race_variables):
    """Sends race JSON object to database for storage"""
    global server, port
    session_id = race_variables['session_id']
    url = '%s:%d/sessions/%d/races' % (server, port, session_id)
    keys = ['start_time', 'duration', 'num_players', 'p1', 'p2', 'p3', 'p4']
    payload = {key : race_variables[key] for key in keys}
    headers = {'content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response
