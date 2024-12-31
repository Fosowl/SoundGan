#!/usr/bin python3

import matplotlib.pyplot as plt
import librosa
import torch
import numpy as np
import cv2
from scipy.io.wavfile import write

def mel_to_waveform(S_dB, sr=22050, n_fft=1024, hop_length=512):
    S_power = librosa.db_to_power(S_dB)
    linear_spectrogram = librosa.feature.inverse.mel_to_stft(S_power, sr=sr, n_fft=n_fft)
    waveform = librosa.griffinlim(linear_spectrogram, hop_length=hop_length, n_iter=256)
    return waveform

def spectrogram_to_wav(img, output_path, sr=22050, hop_length=512, n_fft=1024):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    # Convert to mel spectrogram
    S_dB = (img * 80.0) - 80.0
    waveform = mel_to_waveform(S_dB, sr=sr, n_fft=n_fft, hop_length=hop_length)
    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0)
    max_val = np.max(np.abs(waveform))
    waveform = waveform / max_val if max_val > 0 else waveform
    scaled_waveform = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    scaled_waveform = np.ascontiguousarray(scaled_waveform)
    # Save as WAV
    write(output_path, sr, scaled_waveform)

def inference(device, config):
    netG = torch.load(f"{config.saveroot}/model_G.pt", map_location=device)
    netG = netG.to(device)
    netG.eval()
    b_size = 1
    z = torch.randn(b_size, config.nz, 1, 1, device=device)  # Random latent vector
    imgs = netG(z)
    imgs = imgs.cpu().detach().numpy()
    img = imgs[0]
    img = cv2.resize(img,
                    (config.original_image_size[1], config.original_image_size[0]),
                    interpolation=cv2.INTER_CUBIC)
    spectrogram_to_wav(img[0], "output.wav")
    plt.figure(figsize=(config.original_image_size[0] / 100, config.original_image_size[1] / 100), dpi=100)
    plt.imshow(img[0], cmap='viridis', aspect='auto')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig("output_inference.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()