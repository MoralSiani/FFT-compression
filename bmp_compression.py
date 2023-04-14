import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import fft
import numpy as np
import util
from pathlib import Path


CUTOFF_PERCENT = 0.15
LOG_FACTOR = 0.2


# ### Compression ### #

def compress_and_write(image, output_file, freq_domain=None):
    compressed_freq_domain, padding = compress(image, freq_domain)
    util.np_savez(output_file, compressed_freq_domain, padding)
    return {
        'arr_0': compressed_freq_domain,
        'arr_1': padding,
    }


def compress(image, freq_domain):
    """returns a compressed data"""
    # run fft
    print('Compressing...')
    if freq_domain is None:
        freq_domain = fft.fft2d_with_channels(image)
    # compress and return
    compressed_freq_domain, padding = compress_freq_domain_by_truncating(freq_domain)
    converted_compressed_freq_domain = compressed_freq_domain.astype('complex64')
    print('Compression complete')
    return converted_compressed_freq_domain, np.array(padding)


def compress_freq_domain_by_truncating(freq_domain):
    cutoff = int(CUTOFF_PERCENT * len(freq_domain[0]) / 2)
    compressed_freq_domain = combine_corners(*get_image_corners(freq_domain, cutoff))
    padding = int(len(freq_domain[0]) / 2 - cutoff)
    return compressed_freq_domain, padding


# ### Decompression ### #

def decompress_and_write(compressed_data, output_file):
    image = decompress(compressed_data)
    util.write_bmp_file(output_file, image)
    return image


def decompress(compressed_data):
    """Returns the decompressed data as a bmp file"""
    # Extract data
    compressed_freq_domain = compressed_data['arr_0']
    padding = compressed_data['arr_1'].item()

    # run inverted fft
    freq_domain = pad_freq_domain(compressed_freq_domain, padding)
    print('Decompressing...')
    image = np.real(fft.inverse_fft2d_with_channels(freq_domain))
    print('Decompression complete')
    return normalize_data(image).astype('uint8')


def pad_freq_domain(compressed_freq_domain, padding):
    corner_len = int(len(compressed_freq_domain) / 2)
    image_len = len(compressed_freq_domain) + (2 * padding)
    up_left, up_right, down_left, down_right = get_image_corners(compressed_freq_domain)
    padded_freq_domain = np.zeros(image_len * image_len * 3, dtype=np.complex128).reshape(image_len, image_len, 3)
    padded_freq_domain[0:corner_len, 0:corner_len, :] += up_left
    padded_freq_domain[0:corner_len, -corner_len:image_len, :] += up_right
    padded_freq_domain[-corner_len:image_len, 0:corner_len, :] += down_left
    padded_freq_domain[-corner_len:image_len, -corner_len:image_len, :] += down_right
    return padded_freq_domain


# ### Utils ### #

def normalize_data(data, log_factor=1):
    # returns the component's norm normalized between 0-255 with centered axes.
    data_norms = np.abs(data)
    max_value = np.max(data_norms)
    return ((data_norms / max_value) ** log_factor) * 255


def get_image_corners(arr, size=None):
    if size is None:
        size = int(len(arr) / 2)
    assert size <= len(arr) / 2
    up_left = arr[0:size, 0:size, :]
    up_right = arr[0:size, -size:len(arr), :]
    down_left = arr[-size:len(arr), 0:size, :]
    down_right = arr[-size:len(arr), -size:len(arr), :]
    return up_left, up_right, down_left, down_right


def combine_and_shift_corners(up_left, up_right, down_left, down_right):
    up = np.hstack((down_right, down_left))
    down = np.hstack((up_right, up_left))
    return np.vstack((up, down))


def combine_corners(up_left, up_right, down_left, down_right):
    up = np.hstack((up_left, up_right))
    down = np.hstack((down_left, down_right))
    return np.vstack((up, down))


# ### Analysis ### #

def analyze_bmp(bmp_file, output_dir):
    # data
    image = util.read_bmp_file(bmp_file)
    image_domain = np.array(image)

    # # 2D fft to calculate freq. domain
    freq_domain = fft.fft2d_with_channels(image_domain)
    normalized_freq_domain = combine_and_shift_corners(*get_image_corners(normalize_data(freq_domain, LOG_FACTOR)))

    # Compress and write
    output_file = Path(output_dir / f'{bmp_file.stem}.bcmp')
    compressed_data = compress_and_write(image_domain, output_file, freq_domain)

    # Decompress and write
    padding = compressed_data['arr_1'].item()
    decompressed_freq_domain = pad_freq_domain(compressed_data['arr_0'], padding)
    output_file = output_dir / f'{bmp_file.stem}_modified.bmp'
    decompressed_image = decompress_and_write(compressed_data, output_file)
    normalized_decompressed_freq_domain = combine_and_shift_corners(*get_image_corners(normalize_data(decompressed_freq_domain, LOG_FACTOR)))

    # Plotting
    output_file = output_dir / f'{bmp_file.stem}_analysis.html'
    plot_image(
        image,
        normalized_freq_domain,
        decompressed_image,
        normalized_decompressed_freq_domain,
        output_file
    )


# ### Plotting ### #

def plot_image(image, freq_domain, decompressed_image, decompressed_freq_domain, output_file):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Original image",
            "frequency domain",
            "Decompressed image",
            "Decompressed frequency domain"
        )
    )
    fig.add_trace(go.Image(z=image), row=1, col=1)
    fig.add_trace(go.Image(z=freq_domain), row=1, col=2)
    fig.add_trace(go.Image(z=decompressed_image), row=2, col=1)
    fig.add_trace(go.Image(z=decompressed_freq_domain), row=2, col=2)
    fig.update_layout(
        autosize=True,
        width=1900,
        height=1300,
)
    # # x-axis names
    # fig.update_xaxes(title_text="Time", row=1, col=1)
    # fig.update_xaxes(title_text="Frequency", row=2, col=1)
    #
    # # y-axis names
    # fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    # fig.update_yaxes(title_text="Magnitude", row=2, col=1)

    # Update geo subplot properties
    # fig.update_geos(
    #     projection_type="orthographic",
    #     landcolor="white",
    #     oceancolor="MidnightBlue",
    #     showocean=True,
    #     lakecolor="LightBlue"
    # )

    fig.update_layout(
        template="plotly_dark",
        # title_text="Before compression",
    )
    fig.write_html(output_file)
    os.startfile(output_file)
