import numpy as np
from scipy.io import wavfile as wf
import fft


dtypes = [np.uint8, np.int16, np.int32, np.float32]


def _compress_wav(sampling_rate, time_domain):
    # extract data
    original_time_domain_size = len(time_domain)
    dtype_idx = dtypes.index(time_domain.dtype)
    freq_res = sampling_rate / len(time_domain)
    print('Compressing...')
    # run fft
    time_domain = fft.power2_round_up(time_domain)
    freq_domain = fft.fft(time_domain)
    # compress and save
    compressed_freq_domain, padding = slice_freq_domain(freq_domain, freq_res)
    output_array = np.concatenate(([original_time_domain_size], [sampling_rate],
                                   [dtype_idx], compressed_freq_domain, [padding]))
    print('Compress complete')
    return output_array


def slice_freq_domain(freq_domain, freq_res):
    freq_cutoff = get_freq_cutoff(len(freq_domain), freq_res)
    compressed_freq_domain = freq_domain[0:freq_cutoff]
    padding = len(freq_domain) - freq_cutoff
    return compressed_freq_domain, padding


def get_freq_cutoff(max_cutoff, freq_res):
    return min(max_cutoff, int(3000 / freq_res))


def _decompress(compressed_data):
    # extract data
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
    print('Decompress complete')
    return sampling_rate, time_domain


def compress_wav(wav_file, out_file):
    sampling_rate, time_domain = wf.read(wav_file)
    output_array = _compress_wav(sampling_rate, time_domain)
    np.save(out_file, arr=output_array)


def decompress(compressed_file_path, out_file):
    compressed_data = np.load(compressed_file_path)
    sampling_rate, time_domain = _decompress(compressed_data)
    wf.write(out_file, sampling_rate, time_domain)

