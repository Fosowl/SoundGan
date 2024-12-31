#!/usr/bin python3

from __future__ import unicode_literals
import logging
import os
import sys
import yt_dlp as youtube_dl
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
import openai
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def confirm_download(full_path: str, min_bytes: int) -> bool:
    if os.path.exists(full_path) == False:
        return False
    if os.path.getsize(full_path) < min_bytes:
        return False
    return True

def safe_remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)

def get_yt_options(full_path: str) -> dict:
    return {
        'quiet': False,
        'format': 'worst',
        'outtmpl': full_path,
        'noplaylist': True,
        'continue_dl': True,
        'keepvideo': False,
        'verbose': True,
        'extractor_args': {'youtube': {'nocheckcertificate': True}},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        #'postprocessor_args': ['-t', duration_stamp]
    }

def download_clip_yt(ydl, url: str, info_dict: dict, full_path_wav: str) -> None:
    try:
        ydl.prepare_filename(info_dict)
        ydl.download([url])
    except Exception as e:
        safe_remove(full_path_wav)
        raise e
    
def download_clip(url: str, name: str, path_folder: str, config: dict) -> bool:
    full_path = f'{path_folder}/{name}' 
    full_path_wav = f'{path_folder}/{name}.wav' 
    ydl_opts = get_yt_options(full_path)
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            info_dict = ydl.extract_info(url, download=False)

            if info_dict == None:
                logger.warning(f"Empty info dict for {name}")
                return False
            
            if 'duration' in info_dict:
                duration = info_dict['duration']
            else:
                return False

            if duration > config["MAX_VIDEO_DURATION"]:
                logger.warning(f"Video too long for {name}")
                return False
            if duration > config["SAMPLE_AUDIO_DURATION"]:
                download_clip_yt(ydl, url, info_dict, full_path_wav)
            else:
                logger.warning(f"Video too short for {name}")
                return False
    except Exception as e:
        logger.error(f"Fatal error on download of {name} : {e}")
        return False

    if confirm_download(full_path_wav, 20000) == False:
        logger.warning(f"Download not confirmed {url}")
        safe_remove(full_path_wav)
        return False
    return True

def load_checkpoint_file(saving_file: str) -> str:
    try:
        f = open(saving_file, 'r')
        txt = f.read()
        return txt.split('\n')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {saving_file} not found")
    except Exception as e:
        raise e

def save_download(save_file, url: str) -> None:
    try:
        save_file.write(f"{url}\n")
    except Exception as e:
        raise e

def create_folder_if_not_exists(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Folder {folder_path} created.")

def whisper_check_voices(wav_path: str, max_char: int) -> bool:
    f = open(wav_path, "rb")
    try:
        interpretation = openai.Audio.translate("whisper-1", f, temperature=0)
    except Exception as e:
        logger.error(f"Openai failed to interpret : {e}")
        return False
    if len(interpretation["text"]) > max_char:
        return True
    return False

def download_clip_samples(url: str, name: str, path_folder: str, save_file: str, config: dict) -> bool:
    wav_path = f'{path_folder}/{name}.wav' 
    if download_clip(url, name, path_folder, config) == False:
        safe_remove(wav_path)
        return False
    save_download(save_file, url)
    logger.info(f"Downloaded {name}")
    audio = AudioSegment.from_file(wav_path, format="wav")
    duration = len(audio)
    split_duration = config["SAMPLE_AUDIO_DURATION"]
    i = config["START_SAMPLE_IDX"]
    man_voice_count = 0
    spacing = 2
    while i * split_duration <= (duration/1000) - split_duration:
        start_time = (i-1)*split_duration*1000
        end_time = i*split_duration*1000 
        part = audio[start_time:end_time]
        part_path = f"{path_folder}/{name}_{i}.wav"
        part.export(part_path, format="wav")
        if whisper_check_voices(part_path, 25) == True:
            man_voice_count += 1
            safe_remove(part_path)
        else:
            logger.info(f"extracted {i-config['START_SAMPLE_IDX']}th sample...")
            man_voice_count = 0
        if man_voice_count >= 3:
            spacing *= 3
        i += spacing
    safe_remove(wav_path)
    return True
    
def check_donwloaded(url: str, downloaded: list) -> bool:
    for d in downloaded:
        if url == d: 
            return True
    return False

def downloader(config: dict, class_name: str) -> None:
    path_csv = Path(config["CSV_FOLDER_PATH"]) / f"{class_name}.csv"
    output_folder = Path(config["OUTPUT_FOLDER_PATH"]) / class_name
    output_folder.mkdir(parents=True, exist_ok=True)
    try:
        dat = pd.read_csv(path_csv)
        save_file = open("dl_checkpoint", 'a')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {path_csv} not found")
    downloaded = load_checkpoint_file(config["CHECKPOINT_FILE"])
    count = 0
    for index, row in dat.iterrows():
        t = re.sub(r'[^a-zA-Z]', '', row["title"])
        u = row["url"]
        if check_donwloaded(u, downloaded) == True:
            logger.info(f"Already downloaded : {t}")
            continue
        logger.info(f"Downloading : {t} ({u})")
        if download_clip_samples(u, t, output_folder, save_file, config) == True:
            count += 1
        else:
            logger.error(f"Failed download : {t}")
    logger.info(f"downloaded {count} sound from youtube")