from pathlib import Path
import util
from compression import compress_and_write, decompress_and_write


if __name__ == '__main__':
    # Compress
    file = util.get_file()
    path = Path.cwd()/f'compressed_files'/f'{file.stem}'
    path.parent.mkdir(parents=True, exist_ok=True)
    compress_and_write(file, path)

    # Decompress
    out_path = Path.cwd()/f'wav_files'/f'{file.stem}_modified.wav'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decompress_and_write(f'{path}.npy', out_path)


