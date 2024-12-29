#!/usr/bin python3

from __future__ import unicode_literals
import os
import yt_dlp as youtube_dl
import pandas as pd
from colorama import Fore
from pydub import AudioSegment
import openai
import re

def confirm_download(full_path, min_bytes):
    if os.path.exists(full_path) == False:
        return False
    if os.path.getsize(full_path) < min_bytes:
        return False
    return True

def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)

def get_yt_options(full_path):
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
    
def download_clip(url: str, name: str, path_folder: str, config: dict) -> bool:
    full_path = f'{path_folder}/{name}' 
    full_path_wav = f'{path_folder}/{name}.wav' 
    ydl_opts = get_yt_options(full_path)
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            info_dict = ydl.extract_info(url, download=False)
            if info_dict == None:
                print(Fore.RED, "empty info dict", Fore.WHITE)
                return False
            try:
                duration = info_dict['duration']
            except:
                print(Fore.YELLOW, "video is live stream, skipping", Fore.WHITE)
                return False
            if duration > config["MAX_VIDEO_DURATION"]:
                print(Fore.YELLOW, "video too long", Fore.WHITE)
                return False
            if duration > config["SAMPLE_AUDIO_DURATION"]:
                try:
                    ydl.prepare_filename(info_dict)
                    ydl.download([url])
                except Exception as e:
                    safe_remove(full_path_wav)
                    print(Fore.RED, "--- ERROR ---", Fore.WHITE)
                    print(e)
                    raise e
            else:
                print(Fore.YELLOW, f"too short for download", Fore.WHITE)
                return False
    except Exception as e:
        print(Fore.RED, f"Fatal error on download of {name} : {e}", Fore.WHITE)
        return False
    if confirm_download(full_path_wav, 20000) == False:
        print(Fore.YELLOW, f"Download not confirmed {url}", Fore.WHITE)
        safe_remove(full_path_wav)
        return False
    return True

def progress_function(stream, chunk, bytes_remaining):
   print(f"{bytes_remaining} bytes remaining")

def get_downloaded(saving_file: str) -> str:
    try:
        f = open(saving_file, 'r')
        txt = f.read()
        return txt.split('\n')
    except Exception as e:
        print(Fore.RED, "Failed to recover checkpoint", Fore.WHITE)
        raise e

def save_downloaded(save_file, url: str) -> None:
    try:
        save_file.write(f"{url}\n")
    except Exception as e:
        raise e

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created.")

def has_human_voice(wav_path, max_char):
    f = open(wav_path, "rb")
    try:
        interpretation = openai.Audio.translate("whisper-1", f, temperature=0)
    except Exception as e:
        print(Fore.RED, "Failed to interpret :", Fore.WHITE)
        print(Fore.RED, e, Fore.WHITE)
        return False
    if len(interpretation["text"]) > max_char:
        print(Fore.YELLOW, "Voice found : ", interpretation['text'], Fore.WHITE)
        return True
    return False

def download_clip_samples(url, name, path_folder, save_file, config):
    wav_path = f'{path_folder}/{name}.wav' 
    if download_clip(url, name, path_folder, config) == False:
        safe_remove(wav_path)
        return False
    save_downloaded(save_file, url)
    print(Fore.GREEN, "download success!", Fore.WHITE)
    audio = AudioSegment.from_file(wav_path, format="wav")
    duration = len(audio)
    split_duration = config["SAMPLE_AUDIO_DURATION"]
    i = config["START_SAMPLE_IDX"]
    man_voice_count = 0
    spacing = 2
    while i * split_duration <= (duration/1000) - split_duration:
        print(Fore.GREEN, f"extracting {i-config['START_SAMPLE_IDX']}th sample...", Fore.WHITE)
        start_time = (i-1)*split_duration*1000
        end_time = i*split_duration*1000 
        part = audio[start_time:end_time]
        part_path = f"{path_folder}/{name}_{i}.wav"
        part.export(part_path, format="wav")
        if has_human_voice(part_path, 25) == True:
            man_voice_count += 1
            safe_remove(part_path)
        else:
            print(f"Saving part {part_path}th confirmed")
            man_voice_count = 0
        # video is full of voice, skip
        if man_voice_count >= 3:
            spacing *= 3
        i += spacing
    safe_remove(wav_path)
    return True
    
def check_donwloaded(url, downloaded):
    for d in downloaded:
        if url == d: 
            return True
    return False

def downloader(config, class_name):
    path_folder = config["OUTPUT_FOLDER_PATH"] + '/' + class_name
    path_csv = config["CSV_FOLDER_PATH"] + '/' + class_name + ".csv"
    try:
        dat = pd.read_csv(path_csv)
        save_file = open("dl_checkpoint", 'a')
    except FileNotFoundError as e:
        print(e)
        exit(1)
    create_folder_if_not_exists(path_folder)
    downloaded = get_downloaded(config["SAVE_DOWNLOADED_FILE"])
    count = 0
    for index, row in dat.iterrows():
        t = re.sub(r'[^a-zA-Z]', '', row["title"])
        u = row["url"]
        if check_donwloaded(u, downloaded) == True:
            print(Fore.YELLOW, f"Already downloaded : {t}", Fore.WHITE)
            continue
        print(Fore.GREEN, f"Downloading : {t} ({u})", Fore.WHITE)
        if download_clip_samples(u, t, path_folder, save_file, config) == True:
            count += 1
        else:
            print(Fore.RED, f"Failed download : {t}", Fore.WHITE)
    print(f"downloaded {count} sound from youtube")