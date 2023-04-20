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
    sampling_rate, time_domain = wf.read(file)
    if time_domain.ndim != 1:
        raise ValueError('Stereo not supported, only mono ')
    return sampling_rate, time_domain


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
    rounded_size = 2 ** int(math.log2(len(sample)))
    return sample[:rounded_size]


def power2_round_up(sample):
    """Returns a rounded up zero-padded sample as ndarray to the closest power of 2"""
    rounded_size = 2 ** math.ceil(math.log2(len(sample)))
    padding = [0] * (rounded_size - len(sample))
    return np.concatenate((sample, padding))


def power2_round_up_2d(sample):
    vertical_rounded_size = 2 ** math.ceil(math.log2(sample.shape[0]))
    horizontal_rounded_size = 2 ** math.ceil(math.log2(sample.shape[1]))
    vertical_padding = (vertical_rounded_size - sample.shape[0]) // 2
    horizontal_padding = (horizontal_rounded_size - sample.shape[1]) // 2
    vertical = (vertical_padding, vertical_padding)
    horizontal = (horizontal_padding, horizontal_padding)
    if sample.shape[0] % 2 == 1:
        vertical = (vertical_padding + 1, vertical_padding)
    if sample.shape[1] % 2 == 1:
        horizontal = (horizontal_padding + 1, horizontal_padding)
    padded_sample = np.pad(sample, pad_width=(vertical, horizontal, (0, 0)), mode='constant',)
    return padded_sample


def crop_center(image, vertical_crop, horizontal_crop):
    image_vertical, image_horizontal = image.shape[0:2]
    vertical_start = (image_vertical // 2) - (vertical_crop // 2)
    horizontal_start = (image_horizontal // 2) - (horizontal_crop // 2)
    return image[vertical_start:vertical_start + vertical_crop, horizontal_start:horizontal_start + horizontal_crop]

