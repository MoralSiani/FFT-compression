from scipy.io import wavfile as wf
import numpy as np
from pathlib import Path
import util
import fft


dtypes = [np.uint8, np.int16, np.int32, np.float32]


def decompress(compressed_file_path, out_file):
    # get data
    compressed_data = np.load(compressed_file_path)
    original_time_domain_size = int(np.real(compressed_data[0]))
    sampling_rate = int(np.real(compressed_data[1]))
    dtype = dtypes[int(np.real(compressed_data[2]))]
    padding = int(np.real(compressed_data[-1]))
    print('Decompressing...')
    # run inverted fft
    padded_freq_domain = np.concatenate((compressed_data[3:-1], [0] * padding))
    time_domain = np.real(fft.fft(padded_freq_domain, inverse=True)).astype(dtype)

    # resize and save
    time_domain = time_domain[0:original_time_domain_size]
    wf.write(out_file, sampling_rate, time_domain)
    print('Decompress complete')


def compress_wav(wav_file, out_file):
    # get data
    sampling_rate, time_domain = wf.read(wav_file)
    original_time_domain_size = len(time_domain)
    dtype_idx = dtypes.index(time_domain.dtype)
    freq_res = sampling_rate / len(time_domain)
    print('Compressing...')
    # run fft
    time_domain = fft.power2_round_up(time_domain)
    freq_domain = fft.fft(time_domain)

    # compress and save
    freq_cutoff = get_freq_cutoff(len(freq_domain), freq_res)
    compressed_freq_domain, padding = cut_freq_domain(freq_domain, freq_cutoff)
    output_array = np.concatenate(([original_time_domain_size], [sampling_rate], [dtype_idx], compressed_freq_domain, [padding]))
    np.save(out_file, arr=output_array)
    print('Compress complete')

def cut_freq_domain(freq_domain, freq_cutoff):
    compressed_freq_domain = freq_domain[0:freq_cutoff]
    padding = len(freq_domain) - freq_cutoff
    return compressed_freq_domain, padding


def get_freq_cutoff(max_cutoff, freq_res):
    return min(max_cutoff, int(3000 / freq_res))


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


