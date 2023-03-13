import numpy as np
from scipy.io import wavfile as wf
import fft
import util
import os


dtypes = [np.uint8, np.int16, np.int32, np.float32]


def compress(sampling_rate, time_domain):
    """Returns a compressed array with metadata"""
    # extract metadata
    original_time_domain_size = len(time_domain)
    dtype_idx = dtypes.index(time_domain.dtype)
    freq_res = sampling_rate / len(time_domain)

    # run fft
    print('Compressing...')
    time_domain = fft.power2_round_up(time_domain)
    freq_domain = fft.fft(time_domain)

    # compress_and_write and save
    compressed_freq_domain, padding = compress_freq_domain(freq_domain, freq_res)
    output_array = np.concatenate(([original_time_domain_size], [sampling_rate],
                                   [dtype_idx], compressed_freq_domain, [padding]))
    print('Compression complete')
    return output_array


def compress_freq_domain(freq_domain, freq_res):
    """Compress the freq domain"""
    freq_cutoff = compression_by_truncating(len(freq_domain), freq_res)
    compressed_freq_domain = freq_domain[0:freq_cutoff]
    padding = len(freq_domain) - freq_cutoff
    return compressed_freq_domain, padding


def compression_by_truncating(max_cutoff, freq_res):
    """truncate the frequency domain"""
    return min(max_cutoff, int(3000 / freq_res))


def decompress(compressed_data):
    """Returns the decompressed data"""
    # extract data
    original_time_domain_size = int(np.real(compressed_data[0]))
    sampling_rate = int(np.real(compressed_data[1]))
    dtype = dtypes[int(np.real(compressed_data[2]))]
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))

    # run inverted fft
    print('Decompressing...')
    padded_freq_domain = pad_freq_domain(freq_domain, padding)
    time_domain = np.real(fft.inverse_fft(padded_freq_domain)).astype(dtype)

    # resize and save
    time_domain = time_domain[0:original_time_domain_size]
    print('Decompression complete')
    return sampling_rate, time_domain


def pad_freq_domain(freq_domain, padding):
    return np.concatenate((freq_domain, [0] * padding))


def compress_and_write(wav_file_path, out_file_path):
    """receive a wav file path and writes a compressed file to out_file_path"""
    sampling_rate, time_domain = wf.read(wav_file_path)
    output_array = compress(sampling_rate, time_domain)
    np.save(out_file_path, arr=output_array)


def decompress_and_write(compressed_file_path, out_file_path):
    """receive a compressed file path and write the decompressed wav file"""
    compressed_data = np.load(compressed_file_path)
    sampling_rate, time_domain = decompress(compressed_data)
    wf.write(out_file_path, sampling_rate, time_domain)


def analyze_wav(wav_file_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling_rate, time_domain = util.get_file(wav_file_path)
    time_domain = fft.power2_round_down(time_domain)
    freq_domain = fft.fft(time_domain)
    path = output_dir / f'{wav_file_path.stem} - before compression.html'
    time_graph, freq_graph = fft.get_axes(sampling_rate, time_domain, freq_domain)
    util.plot_time_and_freq(time_graph, freq_graph, path)
    os.startfile(path)

    compressed_data = compress(sampling_rate, time_domain)
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))
    new_freq_domain = pad_freq_domain(freq_domain, padding)
    sampling_rate, new_time_domain = decompress(compressed_data)
    path = output_dir / f'{wav_file_path.stem} - after compression.html'
    time_graph, freq_graph = fft.get_axes(sampling_rate, new_time_domain, new_freq_domain)
    util.plot_time_and_freq(time_graph, freq_graph, path)
    os.startfile(path)

