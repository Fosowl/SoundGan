#!/usr/bin python3

import argparse
import torch
from sources.training import training
from sources.inference import inference
from sources.config_loader import Config

parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true', help='Training mode.')
parser.add_argument('--inference', action='store_true', help='Inference mode.')
args = parser.parse_args()

def main():
    config = Config()
    config.load_config('gan_config.json')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    if args.inference:
        inference(device, config, "output.wav")
    elif args.training:
        training(device, config)
    else:
        print("Please specify training or inference mode. --training or --inference")

if __name__ == "__main__":
    main()