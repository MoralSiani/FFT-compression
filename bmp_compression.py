import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import fft
import numpy as np


# ### Compression ### #

def compress(image):
    """given an data as a ndarray, returns a compressed data"""

    # run fft
    print('Compressing...')
    freq_domain = fft.fft2d_with_channels(image)

    # compress_and_write and save
    print('Compression complete')

    return freq_domain


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


def center_and_normalize_frequencies(frequencies):
    # returns the real component normalized between 0-255 with centered axes.
    freq_real_component = np.abs(frequencies)
    max_value = np.max(freq_real_component)
    return center_axes(((freq_real_component / max_value) ** (1/5)) * 255)


def center_axes(freq_domain):
    freq_domain_len = len(freq_domain)
    h_half_len = int(len(freq_domain[0]) / 2)
    v_half_len = int(len(freq_domain[1]) / 2)
    up_left = freq_domain[0:h_half_len, 0:v_half_len]
    up_right = freq_domain[0:h_half_len, v_half_len:freq_domain_len]
    down_left = freq_domain[h_half_len:freq_domain_len, 0:v_half_len]
    down_right = freq_domain[h_half_len:freq_domain_len, v_half_len:freq_domain_len]
    up = np.hstack((down_right, down_left))
    down = np.hstack((up_right, up_left))
    return np.vstack((up, down))
