import cmath
import math
from numpy import fft as npfft
import numpy as np


# ### 2D FFT ### #

def fft2d_with_channels(image):
    # run 2d fft for each channel
    results = np.apply_over_axes(fft2d, image, axes=-1)
    return results


def fft2d(image, *args):
    # vertical fft
    vertical_freq_domain = vertical_fft2d(image)
    # horizontal fft
    freq_domain = horizontal_fft2d(vertical_freq_domain)
    return freq_domain


def vertical_fft2d(data):
    return np.apply_along_axis(fft, axis=0, arr=data)


def horizontal_fft2d(data):
    return np.apply_along_axis(fft, axis=1, arr=data)


# ### 2D inverse FFT ### #

def inverse_fft2d_with_channels(freq_domain):
    # run 2d inverse fft for each channel
    inverse_results = np.apply_over_axes(inverse_fft2d, freq_domain, axes=-1)
    return inverse_results


def inverse_fft2d(freq_domain, *args):
    # horizontal inverse fft
    horizontal_image_domain = horizontal_inverse_fft2d(freq_domain)
    # vertical inverse fft
    image = vertical_inverse_fft2d(horizontal_image_domain)
    return image


def vertical_inverse_fft2d(data):
    return np.apply_along_axis(inverse_fft, axis=1, arr=data)


def horizontal_inverse_fft2d(data):
    return np.apply_along_axis(inverse_fft, axis=0, arr=data)


# ### 1D FFT ### #


def fft(time_domain):
    """Receive complex/real type time domain, returns complex freq domain"""
    freq_domain = _fft_recursive(time_domain, -1)
    return freq_domain


def inverse_fft(freq_domain):
    """Receive complex type freq domain, returns complex type time domain"""
    time_domain = _fft_recursive(freq_domain, 1)
    return time_domain / len(freq_domain)


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


# ### Testing ### #

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
