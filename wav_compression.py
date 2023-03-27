import numpy as np
from scipy.io import wavfile as wf
import fft
import util
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dtypes = [np.uint8, np.int16, np.int32, np.float32]
FREQUENCY = 'frequency'


# ### compression ### #

def compress_and_write(wav_file_path, out_file_path):
    """receive a wav file path and writes a compressed file to out_file_path"""
    sampling_rate, time_domain = wf.read(wav_file_path)
    output_array = compress(sampling_rate, time_domain)
    np.save(out_file_path, arr=output_array)


def compress(sampling_rate, time_domain):
    """Returns a compressed array with metadata"""
    # extract metadata
    original_time_domain_size = len(time_domain)
    dtype_idx = dtypes.index(time_domain.dtype)
    freq_res = sampling_rate / len(time_domain)

    # run fft
    print('Compressing...')
    rounded_time_domain = util.power2_round_up(time_domain)
    freq_domain = fft.fft(rounded_time_domain)

    # compress and return
    compressed_freq_domain, padding = compress_freq_domain(freq_domain, freq_res)
    compressed_data = np.concatenate(([original_time_domain_size], [sampling_rate],
                                      [dtype_idx], compressed_freq_domain, [padding]))
    print('Compression complete')
    return compressed_data


def compress_freq_domain(freq_domain, freq_res):
    """Compress the freq domain"""
    freq_cutoff = compression_by_truncating(len(freq_domain), freq_res)
    compressed_freq_domain = freq_domain[0:freq_cutoff]
    padding = len(freq_domain) - freq_cutoff
    return compressed_freq_domain, padding


def compression_by_truncating(max_cutoff, freq_res):
    """truncate the frequency domain"""
    return min(max_cutoff, int(3000 / freq_res))


# ### Decompression ### #

def decompress_and_write(compressed_file_path, out_file_path):
    """receive a compressed file path and write the decompressed wav file"""
    compressed_data = np.load(compressed_file_path)
    sampling_rate, time_domain = decompress(compressed_data)
    wf.write(out_file_path, sampling_rate, time_domain)


def decompress(compressed_data):
    """Returns the decompressed data as a wav file"""
    # extract data
    original_time_domain_size = int(np.real(compressed_data[0]))
    sampling_rate = int(np.real(compressed_data[1]))
    dtype = dtypes[int(np.real(compressed_data[2]))]
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))

    # run inverted fft
    print('Decompressing...')
    padded_freq_domain = np.concatenate((freq_domain, [0] * padding))
    time_domain = np.real(fft.inverse_fft(padded_freq_domain)).astype(dtype)

    # resize and save
    time_domain = time_domain[0:original_time_domain_size]
    print('Decompression complete')
    return sampling_rate, time_domain


# ### Analysis ### #

def analyze_wav(wav_file_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Read data
    sampling_rate, time_domain = util.read_file(wav_file_path)
    # fft to calculate freq. domain
    time_domain = util.power2_round_down(time_domain)
    freq_domain = fft.fft(time_domain)
    # Plotting
    time_graph, freq_graph = get_axes_as_df(sampling_rate, time_domain, freq_domain)
    before_compression_html = output_dir / f'{wav_file_path.stem}_before_compression.html'
    plot_time_and_freq(time_graph, freq_graph, before_compression_html)
    os.startfile(before_compression_html)
    # Compression
    compressed_data = compress(sampling_rate, time_domain)
    # Decompression
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))
    new_freq_domain = np.concatenate((freq_domain, [0] * padding))
    sampling_rate, new_time_domain = decompress(compressed_data)
    # Plotting
    time_graph, freq_graph = get_axes_as_df(sampling_rate, new_time_domain, new_freq_domain)
    after_compression_html = output_dir / f'{wav_file_path.stem}_after_compression.html'
    plot_time_and_freq(time_graph, freq_graph, after_compression_html)
    os.startfile(after_compression_html)


def get_padded_frequency_domain(compressed_data):
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))
    padded_freq_domain = np.concatenate((freq_domain, [0] * padding))
    return padded_freq_domain


# ### Plotting ### #

def plot_time_and_freq(time_domain, freq_domain, path):
    fig = make_subplots(
        rows=2, cols=1,
        # column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        specs=[[{"type": "scatter"}],
               [{"type": "scatter"}]],
        subplot_titles=("Time domain", "Frequency domain")
    )
    fig.add_trace(
        go.Scatter(x=time_domain['time'], y=time_domain['magnitude'], mode='lines'), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=freq_domain['frequency'], y=freq_domain['magnitude'], mode='lines'), row=2, col=1)

    # x-axis names
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Frequency", row=2, col=1)

    # y-axis names
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)

    # Update geo subplot properties
    fig.update_geos(
        projection_type="orthographic",
        landcolor="white",
        oceancolor="MidnightBlue",
        showocean=True,
        lakecolor="LightBlue"
    )

    fig.update_layout(
        template="plotly_dark",
        # title_text="Before compression",
    )
    fig.write_html(path)


def get_frequency_bins(freq_domain):
    """Given a frequency domain, returns the frequency bins' amplitudes (y-axis for plotting)"""
    cutoff_fft_result = freq_domain[:int(len(freq_domain) / 2)]
    amplitudes = np.abs(cutoff_fft_result) / len(freq_domain) * 2
    # [(complex_norm(e) * 2) / len(freq_domain) for e in cutoff_fft_result]
    return amplitudes


def get_time_axis(sampling_rate, time_domain):
    return np.linspace(0, len(time_domain) / sampling_rate, len(time_domain))


def get_freq_axis(sampling_rate, freq_domain):
    freq_res = sampling_rate / len(freq_domain)
    return np.arange(0, sampling_rate / 2, freq_res)


def get_axes_as_df(sampling_rate, time_domain, freq_domain):
    """Returns time and frequency domains' x and y values as a dataframe.
    Used for plotting with plotly"""
    x1 = get_time_axis(sampling_rate, time_domain)
    y1 = time_domain
    time_domain_df = pd.DataFrame(np.array([x1, y1]).T, columns=['time', 'magnitude'])

    x2 = get_freq_axis(sampling_rate, freq_domain)
    y2 = get_frequency_bins(freq_domain)
    freq_domain_df = pd.DataFrame(np.array([x2, y2]).T, columns=[FREQUENCY, 'magnitude'])

    return time_domain_df, freq_domain_df
