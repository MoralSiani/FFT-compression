from pathlib import Path
import util
from compression import compress_wav, decompress


if __name__ == '__main__':
    # Compress
    file = util.get_file()
    path = Path.cwd()/f'compressed_files'/f'{file.stem}'
    path.parent.mkdir(parents=True, exist_ok=True)
    compress_wav(file, path)

    # Decompress
    out_path = Path.cwd()/f'wav_files'/f'{file.stem}_modified.wav'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decompress(f'{path}.npy', out_path)


