import cmath
import math
from numpy import fft as npfft
import numpy as np
import matplotlib.pyplot as plt


def fft(time_domain):
    """Receive complex/real type time domain, returns complex freq domain"""
    results = _fft_recursive(time_domain, -1)
    return results


def inverse_fft(freq_domain):
    """Receive complex type freq domain, returns complex type time domain"""
    results = _fft_recursive(freq_domain, 1)
    return results / len(freq_domain)


def _fft_recursive(sample, inverse_coefficient):
    """converts between time and freq domains given an inverse coefficient.
    inverse coefficient is 1 when inversing, otherwise -1"""
    # break condition
    sample_size = len(sample)
    if sample_size == 1:
        return sample

    # assert data is a power of 2
    sample_log = math.log2(sample_size)
    if sample_log != int(sample_log):
        raise ValueError

    # recursion step
    evens = sample[0:-1:2]
    odds = sample[1:sample_size:2]
    feven, fodd = _fft_recursive(evens, inverse_coefficient), _fft_recursive(odds, inverse_coefficient)

    # calculating domain bins
    coeff_const = inverse_coefficient * (2j * cmath.pi) / sample_size
    half_sample_size = int(sample_size / 2)
    domain_bins1 = []
    domain_bins2 = []
    for k in range(half_sample_size):
        f_k = feven[k] + cmath.exp(coeff_const * k) * fodd[k]
        domain_bins1.append(f_k)
        k2 = k + half_sample_size
        f_k2 = feven[k] + cmath.exp(coeff_const * k2) * fodd[k]
        domain_bins2.append(f_k2)
    return np.asarray(domain_bins1 + domain_bins2)


def power2_round_down(sample):
    """Returns a rounded down truncated time_domain to the closest power of 2"""
    cutoff = 2 ** int(math.log2(len(sample)))
    return sample[:cutoff]


def power2_round_up(sample):
    """Returns a rounded up zero-padded time_domain to the closest power of 2"""
    cutoff = 2 ** math.ceil(math.log2(len(sample)))
    padding = [0] * (cutoff - len(sample))
    return np.concatenate((sample, padding))


def get_frequency_bins(freq_domain):
    """Given a frequency domain, returns the frequency bins' amplitudes (y-axis for plotting)"""
    cutoff_fft_result = freq_domain[:int(len(freq_domain) / 2)]
    amplitudes = [(complex_norm(e) * 2) / len(freq_domain) for e in cutoff_fft_result]
    return amplitudes


def complex_norm(complex_num):
    return math.sqrt(complex_num.real ** 2 + complex_num.imag ** 2)


def plot_fft(sampling_rate, time_domain, freq_domain):
    fig, axs = plt.subplots(2)
    fig.suptitle('Wave and Frequency bins')
    x1 = np.linspace(0, len(time_domain) / sampling_rate, len(time_domain))
    y1 = time_domain
    axs[0].plot(x1, y1, linewidth=0.8)

    freq_res = sampling_rate / len(freq_domain)
    x2 = np.arange(0, sampling_rate / 2, freq_res)
    y2 = get_frequency_bins(freq_domain)
    axs[1].plot(x2, y2, linewidth=0.8)

    plt.show()


def test_fft_comparison():
    sample = np.asarray([0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707])
    my_results = fft(sample)
    np_result = npfft.fft(sample)
    eps = 10 ** -5
    assert (np.abs(my_results - np_result) >= eps).sum() == 0


def test_inverse():
    sample = np.asarray([0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707])
    fft_results = fft(sample)
    inverse_results = inverse_fft(fft_results)
    eps = 10 ** -5
    assert (np.abs(sample - inverse_results) >= eps).sum() == 0


test_fft_comparison()
test_inverse()
