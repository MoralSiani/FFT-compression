import numpy as np
import fft
import util
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

dtypes = [np.uint8, np.int16, np.int32, np.float32]


# ### compression ### #

def compress_and_write(wav_file: Path, output_file: Path):
    """receive a wav file output_file and writes a compressed file to out_file_path"""
    sampling_rate, time_domain = util.read_wav_file(wav_file)
    compressed_data = compress(sampling_rate, time_domain)
    util.np_save(output_file, compressed_data)


def compress(sampling_rate, time_domain, freq_domain=None):
    """Returns a compressed array with metadata
    if given freq_domain, compress will not run fft"""
    # extract metadata
    original_time_domain_size = len(time_domain)
    dtype_idx = dtypes.index(time_domain.dtype)
    freq_res = sampling_rate / len(time_domain)

    if freq_domain is None:
        freq_domain = fft.fft(util.power2_round_up(time_domain))

    # compress and return
    print('Compressing...')
    compressed_freq_domain, padding = compress_freq_domain(freq_domain, freq_res)
    compressed_data = np.concatenate((
        [original_time_domain_size],
        [sampling_rate],
        [dtype_idx],
        compressed_freq_domain,
        [padding]
    ))
    print('Compression complete')
    return compressed_data


def compress_freq_domain(freq_domain, freq_res):
    """Compress the freq domain"""
    freq_cutoff = compression_by_truncating(len(freq_domain), freq_res, freq_cutoff=3000)
    compressed_freq_domain = freq_domain[0:freq_cutoff]
    padding = len(freq_domain) - freq_cutoff
    return compressed_freq_domain, padding


def compression_by_truncating(max_cutoff, freq_res, freq_cutoff):
    """truncate the frequency domain"""
    return min(max_cutoff, int(freq_cutoff / freq_res))


# ### Decompression ### #

def decompress_and_write(compressed_wav_file: Path, output_file: Path):
    """receive a compressed file output_file and write the decompressed wav file"""
    compressed_data = np.load(compressed_wav_file)
    sampling_rate, time_domain = decompress(compressed_data)
    util.write_to_wav_file(output_file, sampling_rate, time_domain)


def decompress(compressed_data):
    """Decompresses and returns the decompressed data as a wav file"""
    # Extract data
    original_time_domain_size = int(np.real(compressed_data[0]))
    sampling_rate = int(np.real(compressed_data[1]))
    dtype = dtypes[int(np.real(compressed_data[2]))]
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))

    # Run inverted fft
    print('Decompressing...')
    padded_freq_domain = np.concatenate((freq_domain, [0] * padding))
    time_domain = np.real(fft.inverse_fft(padded_freq_domain)).astype(dtype)

    # Resize and return
    time_domain = time_domain[0:original_time_domain_size]
    print('Decompression complete')
    return sampling_rate, time_domain


# ### Analysis ### #

def analyze_wav(wav_file, output_dir):
    """Given a wav file, calculate and saves an analysis file in output_dir, with time and
    frequency domains' graphs before and after compression and decompression."""
    # Read data
    sampling_rate, input_time_domain = util.read_wav_file(wav_file)

    # fft to calculate freq. domain
    time_domain = util.power2_round_down(input_time_domain)
    freq_domain = fft.fft(time_domain)

    # Compress and save
    compressed_data = compress(sampling_rate, input_time_domain, freq_domain=freq_domain)
    compressed_file = output_dir / f'{wav_file.stem}.wcmp'
    util.np_save(compressed_file, compressed_data)

    # Decompress and save
    compressed_freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))
    decompressed_freq_domain = np.concatenate((compressed_freq_domain, [0]*padding))
    _, decompressed_time_domain = decompress(compressed_data)
    decompressed_file = output_dir / f'{wav_file.stem}_modified.wav'
    util.write_to_wav_file(decompressed_file, sampling_rate, decompressed_time_domain)

    # get graphs and plot
    original_time_graph, original_freq_graph = get_axes_as_df(
        sampling_rate,
        time_domain,
        freq_domain
    )
    decompressed_time_graph, decompressed_freq_graph = get_axes_as_df(
        sampling_rate,
        decompressed_time_domain,
        decompressed_freq_domain,
    )

    analysis_file = output_dir / f'{wav_file.stem}_analysis.html'
    plot_time_and_freq(
        original_time_graph,
        original_freq_graph,
        decompressed_time_graph,
        decompressed_freq_graph,
        analysis_file)

    os.startfile(analysis_file)


def get_padded_frequency_domain(compressed_data):
    freq_domain = compressed_data[3:-1]
    padding = int(np.real(compressed_data[-1]))
    padded_freq_domain = np.concatenate((freq_domain, [0] * padding))
    return padded_freq_domain


def plot_time_and_freq(time_domain, freq_domain, cmp_time_domain, cmp_freq_domain, output_file):
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        specs=[[{"type": "scatter"},
               {"type": "scatter"}],
               [{"type": "scatter"},
               {"type": "scatter"}],
               ],
        subplot_titles=(
            "Time domain",
            "Frequency domain",
            "Compressed_time domain",
            "Compressed_frequency domain"
        )
    )
    fig.add_trace(
        go.Scatter(x=time_domain['time'], y=time_domain['magnitude'], mode='lines'), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=freq_domain['frequency'], y=freq_domain['magnitude'], mode='lines'), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=cmp_time_domain['time'], y=cmp_time_domain['magnitude'], mode='lines'), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=cmp_freq_domain['frequency'], y=cmp_freq_domain['magnitude'], mode='lines'), row=2, col=2
    )
    # x-axis names
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Frequency", row=2, col=2)

    # y-axis names
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2)

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
    fig.write_html(output_file)


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
    freq_domain_df = pd.DataFrame(np.array([x2, y2]).T, columns=['frequency', 'magnitude'])

    return time_domain_df, freq_domain_df
