import os
import json
import argparse

from sources.downloader import downloader
from sources.scrawler import scrawler
from sources.sound2spec import sound2spec

parser = argparse.ArgumentParser()
parser.add_argument('--download', help='Mode for downloading data from youtube.', required=False)
parser.add_argument('--search', help='Mode for searching and selecting sound to download.', required=False)
parser.add_argument('--convert', help='Mode for converting wav to spectrogram.', required=False)
parser.add_argument('--config', help='Path to config file.', required=False)

args = parser.parse_args()

def parse_config(config):
    for key, value in config.items():
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
    return config

def load_config(path):
    try:
        with open(path, 'r') as f:
            config_raw = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {path}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Config file is not a valid JSON file")
    except Exception as e:
        raise e
    config = parse_config(config_raw)
    return config
    
def main():
    config = load_config('config.json')
    if args.download:
        downloader(config, args.download)
    elif args.search:
        scrawler(config, args.search)
    elif args.convert:
        sound2spec(config, args.convert)
    else:
        print("Please specify a mode to run.")

if __name__ == "__main__":
    main()