import util
from wav_compression import compress_and_write, decompress_and_write, analyze_wav
from bmp_compression import compress, decompress, plot_image, center_axes, center_and_normalize_frequencies
from PIL import Image
from pathlib import Path
import numpy as np
import fft


def run_wav_compression():
    file = util.select_wav_file()
    output_dir = Path.cwd() / f'analysis_files'
    analyze_wav(file, output_dir)

    # Compress
    path = Path.cwd() / f'compressed_files' / f'{file.stem}'
    path.parent.mkdir(parents=True, exist_ok=True)
    compress_and_write(file, path)

    # Decompress
    out_path = Path.cwd() / f'wav_files' / f'{file.stem}_modified.wav'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decompress_and_write(f'{path}.npy', out_path)


def run_bmp_compression():
    # data
    file = util.select_bmp_file()
    image = Image.open(file)
    image_domain = np.array(image)

    # 2D fft
    freq_domain = compress(image_domain)
    normalized_freq = center_and_normalize_frequencies(freq_domain)
    centered_normalized_freq = center_axes(center_and_normalize_frequencies(freq_domain))

    # Horizontal 2D fft
    horizontal_freq_domain = fft.horizontal_fft2d(image_domain)
    normalized_horizontal_freq = center_and_normalize_frequencies(horizontal_freq_domain)

    # Vertical 2D fft
    vertical_freq_domain = fft.vertical_fft2d(image_domain)
    normalized_vertical_freq = center_and_normalize_frequencies(vertical_freq_domain)

    # Plot 2D fft
    output_dir = Path.cwd() / f'analysis_files' / f'{file.stem}-forward.html'
    plot_image(image_domain, normalized_freq, normalized_horizontal_freq, normalized_vertical_freq, output_dir)

    # inverse 2D fft
    inverse_image_domain = np.real(decompress(freq_domain))

    # Horizontal inverse 2D fft
    horizontal_image_domain = center_and_normalize_frequencies(fft.horizontal_inverse_fft2d(freq_domain))

    # Vertical inverse 2D fft
    vertical_image_domain = center_and_normalize_frequencies(fft.vertical_inverse_fft2d(freq_domain))

    # Plot inverse 2D fft
    output_dir = Path.cwd() / f'analysis_files' / f'{file.stem}-inverse.html'
    plot_image(inverse_image_domain, normalized_freq, horizontal_image_domain, vertical_image_domain, output_dir)


if __name__ == '__main__':
    run_bmp_compression()

