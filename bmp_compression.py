import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import fft
import numpy as np


# ### Compression ### #

def compress(image):
    """returns a compressed data"""

    # run fft
    print('Compressing...')
    freq_domain = fft.fft2d_with_channels(image)

    # compress and return
    compressed_freq_domain, padding = compress_freq_domain(freq_domain)
    print('Compression complete')

    return compressed_freq_domain, padding


def compress_freq_domain(freq_domain):
    cutoff = int(0.1 * len(freq_domain[0]) / 2)
    padding = len(freq_domain[0]) / 2 - cutoff
    data_mask = np.ones(shape=(cutoff,), dtype=bool)
    padding_mask = np.zeros(shape=(padding,), dtype=bool)
    compression_mask = np.concatenate((data_mask, padding_mask, data_mask))
    return freq_domain[compression_mask, compression_mask]


# ### Decompression ### #

def decompress(compressed_data):
    """Returns the decompressed data as a bmp file"""

    # run inverted fft
    print('Decompressing...')
    image = np.real(fft.inverse_fft2d_with_channels(compressed_data))

    # resize and save
    print('Decompression complete')

    return image


# ### Plotting ### #

def plot_image(image, freq_domain, h_freq, v_freq, output_dir):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Original image", "frequency domain", "Horizontal frequency domain", "Vertical frequency domain")
    )
    fig.add_trace(go.Image(z=image), row=1, col=1)
    fig.add_trace(go.Image(z=freq_domain), row=1, col=2)
    fig.add_trace(go.Image(z=h_freq), row=2, col=1)
    fig.add_trace(go.Image(z=v_freq), row=2, col=2)
    fig.update_layout(
        autosize=True,
        width=1900,
        height=1500,
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
    fig.write_html(output_dir)
    os.startfile(output_dir)


def normalize_frequencies(freq_domain):
    # returns the real component normalized between 0-255 with centered axes.
    freq_real_component = np.abs(freq_domain)
    max_value = np.max(freq_real_component)
    return ((freq_real_component / max_value) ** (1 / 5)) * 255


def shift_axes(up_left, up_right, down_left, down_right):
    up = np.hstack((down_right, down_left))
    down = np.hstack((up_right, up_left))
    return np.vstack((up, down))


def get_quarters(arr, size=None):
    if size is None:
        size = int(len(arr) / 2)
    assert size <= len(arr) / 2
    up_left = arr[0:size, 0:size]
    up_right = arr[0:size, -size:len(arr)]
    down_left = arr[-size:len(arr), 0:size]
    down_right = arr[-size:len(arr), -size:len(arr)]
    return up_left, up_right, down_left, down_right
