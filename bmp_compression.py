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