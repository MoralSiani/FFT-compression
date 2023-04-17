
import os
import argparse
import util
import wav_compression
import bmp_compression
from pathlib import Path
import numpy as np


LOG_FACTOR = 0.2


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='input file (.wav or .bmp)')
    parser.add_argument(
        '-a',
        '--analyze',
        help='compress, decompress and plot graphs (only work with bmp and wav files)',
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
            output_file = Path(output_dir / f'{file.stem}.bcmp')
            image = util.read_bmp_file(file)
            image_arr = np.array(image)
            bmp_compression.compress_and_write(image_arr, output_file)
        case '.bcmp', None:
            output_file = output_dir / f'{file.stem}_modified.bmp'
            compressed_data = np.load(file)
            bmp_compression.decompress_and_write(compressed_data, output_file)
        case '.bmp', True:
            bmp_compression.analyze_bmp(file, output_dir)
        # Not supported options
        case '.wcmp', True:
            raise ValueError('Analysis not supported for compressed files')
        case '.bcmp', True:
            raise ValueError('Analysis not supported for compressed files')
        case _:
            raise ValueError('file not supported')


if __name__ == '__main__':
    run(get_args())
