from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_file():
    files = list((Path.cwd()/'wav_files').iterdir())
    for i, f in enumerate(files):
        print(f'{i+1}.{f.name}')
    return files[int(input('>> ')) - 1]


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
