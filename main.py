from scipy.io import wavfile as wf
import numpy as np
from pathlib import Path
import util
import fft


dtypes = [np.uint8, np.int16, np.int32, np.float32]


def compress_wav(wav_file, out_file):
    sampling_rate, time_domain = wf.read(wav_file)
    print(f'original len: {len(time_domain)}')
    original_time_domain_size = len(time_domain)
    dtype_idx = dtypes.index(time_domain.dtype)
    freq_res = sampling_rate / len(time_domain)
    time_domain = fft.power2_round_up(time_domain)
    freq_domain = fft.fft(time_domain)
    freq_cutoff = get_freq_cutoff(len(freq_domain), freq_res)
    compressed_freq_domain, padding = cut_freq_domain(freq_domain, freq_cutoff)
    output_array = np.concatenate(([original_time_domain_size], [sampling_rate], [dtype_idx], compressed_freq_domain, [padding]))
    np.save(out_file, arr=output_array)


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
    # inverse_results = np.real(fft.fft(fft_results, inverse=True)).astype(np.int16)

