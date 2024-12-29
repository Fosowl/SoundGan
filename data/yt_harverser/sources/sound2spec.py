#!/usr/bin python3

import os
import numpy as np
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wave

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class audio:
    def __init__(self, filepath, hop_length=512, sample_rate=22050):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.waveform, _ = librosa.load(filepath, sr=sample_rate)

    def generate_mel_spectrogram(self, waveform, sr, hop_length=512, n_fft=1024):
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

    def write_disk_spectrogram(self, save_path):
        S_dB = self.generate_mel_spectrogram(self.waveform, self.sample_rate, hop_length=self.hop_length)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=self.sample_rate, hop_length=self.hop_length, cmap='viridis')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved mel spectrogram at {save_path}")

def sound_file_to_spectrogram(full_path, sub_target_folder, idx, name):
        if os.stat(full_path).st_size == 0:
            return
        print("Loading sound : ", full_path)
        sample = audio(full_path)
        img_name = f"sound_{idx}"
        path = sub_target_folder + "/" + img_name
        print(f"writing {path}")
        sample.write_disk_spectrogram(path)

def make_spectral_dataset(sub_input_folder: str,
                          sub_target_folder: str,
                          max_samples_count: int, config: dict) -> None:
    print("Checking folder : ", sub_input_folder)
    files = os.listdir(sub_input_folder)
    idx = 0
    for f in files:
        if idx > max_samples_count:
            print("Reached maximum samples count of : ", max_samples_count)
            break
        if f.endswith('.wav') != True or len(f.split('.')) > 2:
            print(f"Skipping file : {f}")
            continue
        full_path = os.path.join(sub_input_folder, f)
        with wave.open(full_path, 'rb') as audio_file:
            duration = audio_file.getnframes() / audio_file.getframerate()
            if duration == config["SAMPLE_AUDIO_DURATION"]:
                sound_file_to_spectrogram(full_path, sub_target_folder, idx, f[:-4])
        idx += 1

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created.")

def sound2spec(config, class_name, max_count=1000):
    in_path = config["SOUND_FOLDER"] + class_name
    out_path = config["IMAGE_FOLDER"] + class_name
    create_folder_if_not_exists(out_path)
    make_spectral_dataset(in_path, out_path, max_count)