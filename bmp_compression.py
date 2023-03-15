import plotly.express as px
import os
import fft
import numpy as np


def compress(image):
    """given an data as a ndarray, returns a compressed data"""

    # run fft
    print('Compressing...')
    freq_domain = fft.fft2d_with_channels(image)

    # compress_and_write and save
    print('Compression complete')

    return freq_domain


def decompress(compressed_data):
    """Returns the decompressed data as a bmp file"""

    # run inverted fft
    print('Decompressing...')
    image = np.real(fft.inverse_fft2d_with_channels(compressed_data))

    # resize and save
    print('Decompression complete')

    return image


def show_ndarray_image(ndarray_image, output_dir, *, x_label="", y_label=""):
    fig = px.imshow(ndarray_image, labels=dict(x=x_label, y=y_label))
    fig.write_html(output_dir)
    os.startfile(output_dir)
