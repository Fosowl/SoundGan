#!/bin/bash

echo "Running data pipeline for < $1 >"


mkdir -p data/prepared_data/csv
touch data/prepared_data/csv/$1.csv
cd data/data_harverser
python3 ./main.py --scrawl $1 --config config.json
python3 ./main.py --download $1 --config config.json
python3 ./main.py --sound2spec $1 --config config.json