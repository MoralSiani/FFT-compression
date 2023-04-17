import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import fft
import numpy as np
import util
from pathlib import Path


CUTOFF_PERCENT = 0.3
LOG_FACTOR = 0.2


# ### Compression ### #

def compress_and_write(image, output_file, freq_domain=None):
    compressed_freq_domain, padding, image_original_size = compress(image, freq_domain)
    util.np_savez(output_file, compressed_freq_domain, padding, image_original_size)
    return {
        'arr_0': compressed_freq_domain,
        'arr_1': padding,
        'arr_2': image_original_size,
    }


def compress(image, freq_domain):
    """returns a compressed data"""
    # run fft
    print('Compressing...')
    image_original_size = image.shape[0:2]
    if freq_domain is None:
        freq_domain = fft.fft2d_with_channels(util.power2_round_up_2d(image))
    # compress and return
    compressed_freq_domain, padding = compress_freq_domain_by_truncating(freq_domain)
    converted_compressed_freq_domain = compressed_freq_domain.astype('complex64')
    print('Compression complete')
    return converted_compressed_freq_domain, np.array(padding), image_original_size


def compress_freq_domain_by_truncating(freq_domain):
    vertical_cutoff = int(CUTOFF_PERCENT * freq_domain.shape[0] / 2)
    horizontal_cutoff = int(CUTOFF_PERCENT * freq_domain.shape[1] / 2)
    compressed_freq_domain = combine_corners(*get_image_corners(freq_domain, vertical_cutoff, horizontal_cutoff))
    vertical_padding = int(freq_domain.shape[0] / 2 - vertical_cutoff)
    horizontal_padding = int(freq_domain.shape[1] / 2 - horizontal_cutoff)
    return compressed_freq_domain, list((vertical_padding, horizontal_padding))


# ### Decompression ### #

def decompress_and_write(compressed_data, output_file):
    image = decompress(compressed_data)
    util.write_bmp_file(output_file, image)
    return image


def decompress(compressed_data):
    """Returns the decompressed data as a bmp file"""
    # Extract data
    compressed_freq_domain = compressed_data['arr_0']
    padding = compressed_data['arr_1']
    image_original_size = compressed_data['arr_2']
    # run inverted fft
    freq_domain = pad_freq_domain(compressed_freq_domain, padding)
    print('Decompressing...')
    image = np.real(fft.inverse_fft2d_with_channels(freq_domain))
    resized_image = util.crop_center(image, *image_original_size)
    print('Decompression complete')
    return normalize_data(resized_image).astype('uint8')


def pad_freq_domain(compressed_freq_domain, padding):
    corner_vertical_len = int(compressed_freq_domain.shape[0] / 2)
    corner_horizontal_len = int(compressed_freq_domain.shape[1] / 2)
    image_vertical_len = compressed_freq_domain.shape[0] + (2 * padding[0])
    image_horizontal_len = compressed_freq_domain.shape[1] + (2 * padding[1])
    up_left, up_right, down_left, down_right = get_image_corners(compressed_freq_domain)
    padded_freq_domain = np.zeros(image_vertical_len * image_horizontal_len * 3, dtype=np.complex64).reshape(image_vertical_len, image_horizontal_len, 3)
    padded_freq_domain[0:corner_vertical_len, 0:corner_horizontal_len, :] += up_left
    padded_freq_domain[0:corner_vertical_len, -corner_horizontal_len:image_horizontal_len, :] += up_right
    padded_freq_domain[-corner_vertical_len:image_vertical_len, 0:corner_horizontal_len, :] += down_left
    padded_freq_domain[-corner_vertical_len:image_vertical_len, -corner_horizontal_len:image_horizontal_len, :] += down_right
    return padded_freq_domain


# ### Utils ### #

def normalize_data(data, log_factor=1):
    # returns the component's norm normalized between 0-255 with centered axes.
    data_norms = np.abs(data)
    max_value = np.max(data_norms)
    return ((data_norms / max_value) ** log_factor) * 255


def get_image_corners(image, vertical_size=None, horizontal_size=None):
    if vertical_size is None:
        vertical_size = int(image.shape[0] / 2)
    if horizontal_size is None:
        horizontal_size = int(image.shape[1] / 2)
    up_left = image[0:vertical_size, 0:horizontal_size, :]
    up_right = image[0:vertical_size, -horizontal_size:image.shape[1], :]
    down_left = image[-vertical_size:image.shape[0], 0:horizontal_size, :]
    down_right = image[-vertical_size:image.shape[0], -horizontal_size:image.shape[1], :]
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
    freq_domain = fft.fft2d_with_channels(util.power2_round_up_2d(image_domain))
    normalized_freq_domain = combine_and_shift_corners(*get_image_corners(normalize_data(freq_domain, LOG_FACTOR)))

    # Compress and write
    output_file = Path(output_dir / f'{bmp_file.stem}.bcmp')
    compressed_data = compress_and_write(image_domain, output_file, freq_domain)

    # Decompress and write
    padding = compressed_data['arr_1']
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
        rows=1, cols=4,
        subplot_titles=(
            "Original image",
            "frequency domain",
            "Decompressed image",
            "Decompressed frequency domain"
        )
    )
    fig.add_trace(go.Image(z=image), row=1, col=1)
    fig.add_trace(go.Image(z=freq_domain), row=1, col=2)
    fig.add_trace(go.Image(z=decompressed_image), row=1, col=3)
    fig.add_trace(go.Image(z=decompressed_freq_domain), row=1, col=4)
    fig.update_layout(
        autosize=True,
        width=1900,
        height=600,
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

    # fig.update_layout(
    #     template="plotly_dark",
        # title_text="Before compression",
    # )
    fig.update_layout(showlegend=False)
    fig.write_html(output_file)
    os.startfile(output_file)
