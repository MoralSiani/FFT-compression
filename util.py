from pathlib import Path


def get_file():
    files = list((Path.cwd()/'wav_files').iterdir())
    for i, f in enumerate(files):
        print(f'{i+1}.{f.name}')
    return files[int(input('>> ')) - 1]
