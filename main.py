import argparse
import util
import wav_compression
import bmp_compression
from PIL import Image
from pathlib import Path
import numpy as np
import fft


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
    parser.add_argument('file', help='input file (.wav or .bmp)')
    parser.add_argument(
        '-a',
        '--analyze',
        help='compress, decompress and plot graphs (only work with bmp and wac files)',
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument('-o', '--output-dir', help='Output directory [default: data]', default='data')
    parser.add_argument('-c', '--clear', help='Clear output directory', action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def run(args):
    print(args)
    file = Path(args.file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for f in list(output_dir.iterdir()):
            os.remove(f)
            print('Cleared output directory')
    match file.suffix, args.analyze:
        # wav file options
        case '.wav', None:
            output_file = Path(output_dir / f'{file.stem}.wcmp')
            sampling_rate, time_domain = util.read_wav_file(file)
            wav_compression.compress_and_write(sampling_rate, time_domain, output_file)
        case '.wav', True:
            wav_compression.analyze_wav(file, output_dir)
        case '.wcmp', None:
            compressed_data = np.load(file)
            output_file = output_dir / f'{file.stem}_modified.wav'
            wav_compression.decompress_and_write(compressed_data, output_file)

        # bmp file options
        case '.bmp', None:
            pass
        case '.bmp', True:
            pass
        case '.bcmp', None:
            output_file = output_dir / f'{file.stem}_modified.bmp'

        # Not supported options
        case '.wcmp', True:
            raise ValueError('Analysis not supported for compressed files')
        case '.bcmp', True:
            raise ValueError('Analysis not supported for compressed files')
        case _:
            raise ValueError('file not supported')


if __name__ == '__main__':
    run(get_args())
