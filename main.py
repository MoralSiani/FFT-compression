from scipy.io import wavfile as wf
import numpy as np
from pathlib import Path
import util
import fft


def compress_wav(wav_file, out_file):
    sampling_rate, sample = wf.read(wav_file)
    freq_res = sampling_rate / len(sample)
    sample = fft.power2_round_up(sample)
    freq_domain = fft.fft(sample)
    freq_cutoff = get_freq_cutoff(len(freq_domain), freq_res)
    compressed_freq_domain, padding = compress_freq_domain(freq_domain, freq_cutoff)
    compressed_len = len(compressed_freq_domain)
    output_array = np.concatenate((compressed_freq_domain, [padding], [sampling_rate]))
    print(output_array.shape)
    # output_array = [compressed_freq_domain, padding, sampling_rate]

    np.save(out_file, arr=output_array)


def compress_freq_domain(freq_domain, freq_cutoff):
    compressed_freq_domain = freq_domain[0:freq_cutoff]
    padding = len(freq_domain) - freq_cutoff
    return compressed_freq_domain, padding


def get_freq_cutoff(max_cutoff, freq_res):
    return min(max_cutoff, int(3000 / freq_res))


if __name__ == '__main__':
    file = util.get_file()
    path = Path.cwd()/f'wav_files'/f'{file.name}'
    compress_wav(file, path)
    # inverse_results = np.real(fft.fft(fft_results, inverse=True)).astype(np.int16)

