import os
import json
import argparse

from sources.downloader import downloader
from sources.scrawler import scrawler
from sources.sound2spec import sound2spec

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='mode to run', required=True)

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
    if args.mode == 'download':
        downloader(config, "bird")
    elif args.mode == 'scrawl':
        scrawler(config, "bird")
    elif args.mode == 'sound2spec':
        sound2spec(config, "bird")
    else:
        print("Please specify a mode to run.")

if __name__ == "__main__":
    main()