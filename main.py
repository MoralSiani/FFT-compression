from scipy.io import wavfile as wf
import numpy as np
from pathlib import Path
import util
import fft

if __name__ == '__main__':
    file = util.get_file()
    print(file)
    sampling_rate, sample = wf.read(file)
    print(f'sample dtype: {type(sample.dtype)}')
    sample = fft.power2_round_down(sample)
    fft_results = fft.fft(sample)
    fft.plot_fft(sampling_rate, sample, fft_results)
    inverse_results = np.real(fft.fft(fft_results, inverse=True)).astype(np.int16)
    print(sample[:10])
    print(inverse_results[:10])
    path = Path.cwd()/'wav_files'/'test.wav'
    wf.write(path, sampling_rate, inverse_results)
