#!/usr/bin/env python
import requests
import json
import os
import subprocess
import sys
import time


def usage():
    print "Usage: %s <input FLAC file>" % (sys.argv[0])

def detect(input):
    flac = {'file' : open(input)}
    headers = {'Content-type':'audio/x-flac;rate=16000'}
    language = 'en-US'
    parameters = {'xjerr':'1','client':'chromium','lang':language}
    url = 'https://www.google.com/speech-api/v1/recognize'
    req = requests.post(url,
                        params=parameters,
                        headers=headers,
                        files=flac)
    if __name__ == '__main__':
        print "Got data:"
        print req.text
    return req

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
        exit(-1)
    else:
        detect(input=sys.argv[1])
