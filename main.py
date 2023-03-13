import os
from pathlib import Path
import util
from compression import compress_and_write, decompress_and_write, analyze_wav


if __name__ == '__main__':
    file = util.select_file()
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


