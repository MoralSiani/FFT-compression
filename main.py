import util
from wav_compression import compress_and_write, decompress_and_write, analyze_wav
from bmp_compression import show_ndarray_image, compress, decompress
from PIL import Image
from pathlib import Path
import numpy as np
import fft


def normalize_frequencies(frequencies):
    freq_real_component = np.abs(frequencies)
    max_value = np.max(freq_real_component)
    return ((freq_real_component / max_value) ** (1/5)) * 255


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
    print(image_domain.shape)
    print(image_domain.dtype)
    show_ndarray_image(image_domain, Path.cwd() / f'analysis_files' / f'{file.stem}-original.html')

    # Horizontal 2D fft
    horizontal_freq_domain = fft.horizontal_fft2d(image_domain)
    normalized_horizontal_freq = normalize_frequencies(horizontal_freq_domain)
    show_ndarray_image(normalized_horizontal_freq, Path.cwd() / f'analysis_files' / f'{file.stem}-Hfreq.html')

    # Vertical 2D fft
    vertical_freq_domain = fft.vertical_fft2d(image_domain)
    normalized_vertical_freq = normalize_frequencies(vertical_freq_domain)
    show_ndarray_image(normalized_vertical_freq, Path.cwd() / f'analysis_files' / f'{file.stem}-Vfreq.html')

    # 2D fft
    freq_domain = compress(image_domain)
    normalized_freq = normalize_frequencies(freq_domain.copy())
    show_ndarray_image(normalized_freq, Path.cwd() / f'analysis_files' / f'{file.stem}-freq.html')

    # inverse 2D fft
    image_domain = np.real(decompress(freq_domain))
    show_ndarray_image(image_domain, Path.cwd() / f'analysis_files' / f'{file.stem}-inverse.html')


if __name__ == '__main__':
    run_bmp_compression()

