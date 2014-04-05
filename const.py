"""Contains AWS constant fields"""

AWS_ACCESS_KEY_ID = 'AKIAJGDLIAIBWAHY43RQ'
AWS_SECRET_ACCESS_KEY = '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'

# Queue and bucket names that we care about
QUEUES = ['process-queue', 'split-queue']
BUCKETS= ['session-videos', 'race-videos', 'race-audio']

SESSION_BUCKET_BASE = 'https://session-videos.s3.amazonaws.com/'
RACE_BUCKET_BASE = 'https://race-videos.s3.amazonaws.com/'
AUDIO_BUCKET_BASE = 'https://race-audio.s3.amazonaws.com/'

WAIT = 5