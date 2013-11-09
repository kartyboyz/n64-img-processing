# Event detection API

This repository holds the event detection API for the n64 analytics engine.

It is responsible for receiving video, dispatching info for splitting sessions into races, and detecting events within each race.

Not everything works, currently.

## Features

This API offers the following features:

* Start time detection
* End time detection
* Individual race time detection
* Lap time detection
* Item receipt detection
* Collision detection (shells, bananas, and fake item boxes)
* Drift boost detection

Phase 0 --------- Send start time and race duration to server for splitting.

Phase 1 --------- Extract number of players, map, and player characters.

Phase 2 --------- General event detection by race.

Phase 3 --------- Send race statistics off to database server.

Right now, only phase 0 works.
