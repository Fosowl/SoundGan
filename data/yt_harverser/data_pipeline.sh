#!/bin/bash

python3 main.py --scrawl $1
python3 main.py --download $1
python3 main.py --sound2spec $1