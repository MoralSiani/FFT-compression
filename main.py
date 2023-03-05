import os
from pathlib import Path
import util
from compression import compress_and_write, decompress_and_write, compress, decompress, padd_freq_domain
from scipy.io import wavfile as wf
import fft
import numpy as np


def analyze(wav_file_path):
    sampling_rate, time_domain = wf.read(wav_file_path)
    time_domain = fft.power2_round_down(time_domain)
    freq_domain = fft.fft(time_domain)
    path = Path.cwd() / f'analysis_files' / f'{file.stem}.html'
    path.parent.mkdir(parents=True, exist_ok=True)
    time_graph, freq_graph = fft.get_axes(sampling_rate, time_domain, freq_domain)
    util.plot_time_and_freq(time_graph, freq_graph, path)
    os.startfile(path)

    compressed_data = compress(sampling_rate, time_domain)
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))
    new_freq_domain = padd_freq_domain(freq_domain, padding)
    sampling_rate, new_time_domain = decompress(compressed_data)
    path = Path.cwd() / f'analysis_files' / f'{file.stem}2.html'
    path.parent.mkdir(parents=True, exist_ok=True)
    time_graph, freq_graph = fft.get_axes(sampling_rate, new_time_domain, new_freq_domain)
    util.plot_time_and_freq(time_graph, freq_graph, path)
    os.startfile(path)


if __name__ == '__main__':
    file = util.get_file()
    analyze(file)


def run(file):
    # Compress
    path = Path.cwd() / f'compressed_files' / f'{file.stem}'
    path.parent.mkdir(parents=True, exist_ok=True)
    compress_and_write(file, path)

    # Decompress
    out_path = Path.cwd() / f'wav_files' / f'{file.stem}_modified.wav'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decompress_and_write(f'{path}.npy', out_path)


