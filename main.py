import argparse

import util
from wav_compression import compress_and_write, decompress_and_write, analyze_wav
from bmp_compression import compress, decompress, plot_image, shift_axes, normalize_frequencies
from PIL import Image
from pathlib import Path
import numpy as np
import fft


def run_wav_compression(file_name):
    # file = util.select_wav_file()
    print(f'{file_name = }')
    file_path = parse_file(file_name)
    print(f'{file_path = }')
    output_dir = Path.cwd() / f'analysis_files'
    analyze_wav(file_path, output_dir)

    # Compress
    path = Path.cwd() / f'compressed_files' / f'{file_path.stem}'
    path.parent.mkdir(parents=True, exist_ok=True)
    compress_and_write(file_path, path)

    # Decompress
    out_path = Path.cwd() / f'wav_files' / f'{file_path.stem}_modified.wav'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decompress_and_write(f'{path}.npy', out_path)


def run_bmp_compression(file):
    # data
    # file = util.select_bmp_file()
    image = Image.open(file)
    image_domain = np.array(image)

    # 2D fft
    freq_domain, padding = compress(image_domain)
    normalized_freq = normalize_frequencies(freq_domain)

    # Horizontal 2D fft
    horizontal_freq_domain = fft.horizontal_fft2d(image_domain)
    normalized_horizontal_freq = normalize_frequencies(horizontal_freq_domain)

    # Vertical 2D fft
    vertical_freq_domain = fft.vertical_fft2d(image_domain)
    normalized_vertical_freq = normalize_frequencies(vertical_freq_domain)

    # Plot 2D fft
    output_dir = Path.cwd() / f'analysis_files' / f'{file.stem}-forward.html'
    plot_image(image_domain, normalized_freq, normalized_horizontal_freq, normalized_vertical_freq, output_dir)

    # inverse 2D fft
    padding = int(padding)
    padded_freq_domain = np.pad(freq_domain, padding)
    print(padded_freq_domain.shape)
    inverse_image_domain = np.real(decompress(freq_domain))

    # Horizontal inverse 2D fft
    horizontal_image_domain = normalize_frequencies(fft.horizontal_inverse_fft2d(freq_domain))

    # Vertical inverse 2D fft
    vertical_image_domain = normalize_frequencies(fft.vertical_inverse_fft2d(freq_domain))

    # Plot inverse 2D fft
    output_dir = Path.cwd() / f'analysis_files' / f'{file.stem}-inverse.html'
    plot_image(inverse_image_domain, normalized_freq, horizontal_image_domain, vertical_image_domain, output_dir)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wav', help='wav file to compress')
    parser.add_argument('-b', '--bmp', help='bmp file to compress')
    return parser.parse_args()


def parse_file(file_name):
    if args.wav.endswith('wav'):
        file_path = Path.cwd() / f'{file_name}'
    elif args.wav.endswith('bmp'):
        file_path = Path.cwd() / f'{file_name}'
    else:
        raise ValueError('file not supported')
    return file_path


if __name__ == '__main__':
    args = get_args()
    print(args.wav)
    run_wav_compression(args.wav)

