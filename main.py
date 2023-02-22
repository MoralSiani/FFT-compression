import cmath
import math
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft as npfft


def _fft_recursive(sample, inverse_coefficient):
    """takes sample list, returns frequency bins"""
    # breaking condition
    sample_size = len(sample)
    if sample_size == 1:
        return sample

    # assert data is a power of 2
    sample_log = math.log2(sample_size)
    assert sample_log == int(sample_log)

    # recursion step
    evens = sample[0:-1:2]
    odds = sample[1:sample_size:2]
    feven, fodd = _fft_recursive(evens, inverse_coefficient), _fft_recursive(odds, inverse_coefficient)

    # calculating frequncy bins
    coeff_const = inverse_coefficient * (2j * cmath.pi) / sample_size
    half_sample_size = int(sample_size / 2)
    freq_bins1 = []
    freq_bins2 = []
    for k in range(half_sample_size):
        f_k = feven[k] + cmath.exp(coeff_const * k) * fodd[k]
        freq_bins1.append(f_k)
        k2 = k + half_sample_size
        f_k2 = feven[k] + cmath.exp(coeff_const * k2) * fodd[k]
        freq_bins2.append(f_k2)
    return np.asarray(freq_bins1 + freq_bins2)


def fft(sample, inverse=False):
    sample = power2_cutoff(sample, down=False)
    results = _fft_recursive(sample, 1 if inverse else -1)
    if inverse:
        results = results / len(sample)
    return results


def get_frequency_bins(fft_result):
    cutoff_fft_result = fft_result[:int(len(fft_result) / 2)]
    amplitudes = [(complex_norm(e) * 2) / len(fft_result) for e in cutoff_fft_result]
    return amplitudes


def plot(sampling_rate, data_pts, fft_results):
    fig, axs = plt.subplots(2)
    fig.suptitle('Wave and Frequency bins')
    x1 = np.linspace(0, len(data_pts) / sampling_rate, len(data_pts))
    y1 = data_pts
    axs[0].plot(x1, y1, linewidth=0.8)

    freq_res = sampling_rate / len(fft_results)
    x2 = np.arange(0, sampling_rate / 2, freq_res)
    y2 = get_frequency_bins(fft_results)
    axs[1].plot(x2, y2, linewidth=0.8)

    plt.show()


def power2_cutoff(sample, down=True):
    cutoff_log = math.log2(len(sample))
    cutoff_log = int(cutoff_log) if down else math.ceil(cutoff_log)
    cutoff = 2 ** cutoff_log
    return sample[:cutoff]


def complex_norm(complex_num):
    return math.sqrt(complex_num.real ** 2 + complex_num.imag ** 2)


def test_fft_comparison():
    sample = np.asarray([0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707])
    my_results = fft(sample)
    np_result = npfft.fft(sample)
    eps = 10 ** -5
    assert (np.abs(my_results - np_result) >= eps).sum() == 0


def test_inverse():
    sample = np.asarray([0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707])
    fft_results = fft(sample)
    inverse_results = fft(fft_results, inverse=True)
    eps = 10 ** -5
    assert (np.abs(sample - inverse_results) >= eps).sum() == 0


def testing():
    test_fft_comparison()
    test_inverse()
    print('----- All tests passed -----')


def f(t):
    pi = math.pi
    expr1 = 2 * math.cos(pi/2 * t - pi)
    expr2 = 3 * math.cos(pi * t)
    return 5 + expr1 + expr2


if __name__ == '__main__':
    testing()
    sampling_rate, sample = wf.read('wav_files\\speech_modified.wav')
    fft_results = fft(sample)
    plot(sampling_rate, sample, fft_results)
