#!/bin/bash

echo "Running data pipeline for < $1 >"


mkdir -p data/prepared_data/csv
touch data/prepared_data/csv/$1.csv
python3 data/data_harverser/main.py --scrawl $1 --config data/data_harverser/config.json
python3 data/data_harverser/main.py --download $1 --config data/data_harverser/config.json
python3 data/data_harverser/main.py --sound2spec $1 --config data/data_harverser/config.json