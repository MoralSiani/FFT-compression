from pathlib import Path
from scipy.io import wavfile as wf
import math
import numpy as np
import os
from PIL import Image


def select_wav_file():
    files = list((Path.cwd()/'wav_files').iterdir())
    for i, f in enumerate(files):
        print(f'{i+1}.{f.name}')
    return files[int(input('>> ')) - 1]


def select_bmp_file():
    files = list((Path.cwd()/'bmp_files').iterdir())
    for i, f in enumerate(files):
        print(f'{i+1}.{f.name}')
    return files[int(input('>> ')) - 1]


def read_wav_file(file):
    return wf.read(file)


def write_wav_file(file_path, sampling_rate, time_domain):
    wf.write(file_path, sampling_rate, time_domain)


def read_bmp_file(file):
    return Image.open(file)


def write_bmp_file(file, data):
    Image.fromarray(data).save(file)


def np_save(file: Path, data):
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, data)
    if os.path.exists(file):
        os.remove(file)
    os.rename(f'{file}.npy', file)


def np_savez(file: Path, *args):
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file, *args)
    if os.path.exists(file):
        os.remove(file)
    os.rename(f'{file}.npz', file)


def power2_round_down(sample):
    """Returns a rounded down truncated sample to the closest power of 2"""
    cutoff = 2 ** int(math.log2(len(sample)))
    return sample[:cutoff]


def power2_round_up(sample):
    """Returns a rounded up zero-padded sample as ndarray to the closest power of 2"""
    cutoff = 2 ** math.ceil(math.log2(len(sample)))
    padding = [0] * (cutoff - len(sample))
    return np.concatenate((sample, padding))



