import pandas as pd
import numpy as np
import soundfile as sf
import pathlib
import scipy.signal as sig
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils import *


# Filtering thresholds for SNRL, SNR8, SNR9, AL, A8, A9, SL, S8, S9 (in order)
TH = [4, 0, 0, 0, 0.3, 0.35, 0, 0, 0]


def detect_chorus(survey_path, save_path):
    """
    :param survey_path: path to folder (can have subfolders) wich contains your audio data - edit to execute
    :return:
    """
    # Make sure the survey path is in the correct format
    if not isinstance(survey_path, pathlib.Path):
        survey_path = pathlib.Path(survey_path)
    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)
    # Create empty table to store values
    # Set variable names of table
    # SNRL = SNR of LFC2
    # SNR8 = SNR of HFC8
    # SNR9 = SNR of HFC9
    # AL = Area of LFC2
    # A8 = Area of HFC8
    # A9 = Area of HFC9
    # SL = Slope of LFC2
    # S8 = Slope of HFC8
    # S9 = Slope of HFC9
    parameters = pd.DataFrame(columns=['file', 'SNRL', 'SNR8', 'SNR9', 'AL', 'A8', 'A9', 'SL', 'S8', 'S9'])

    # Loop through wav files to compute metrics
    for file_path in tqdm(survey_path.glob('*.wav')):
        # Get sampling rate and duration info from wave file
        sound_file = sf.SoundFile(file_path)
        fs = sound_file.samplerate

        # Read and normalize wave file
        x = sound_file.read()
        x = stats.zscore(x)

        # If the sampling rate is not 500, downsample
        if fs > 500:
            lcm = np.lcm(int(fs), int(500))
            ratio_up = int(lcm / fs)
            ratio_down = int(lcm / 500)
            lowpass_filt = sig.butter(4, 500/2, btype='low', fs=fs, output='sos')
            x = sig.sosfilt(lowpass_filt, x)
            x = sig.resample_poly(x, ratio_up, ratio_down)
            fs = 500

        # Calculate PSD (p) of wav file and extract frequency resolution (f)
        nfft = 2048
        noverlap = int(nfft/2)
        window = sig.windows.hamming(nfft)
        f, p = sig.welch(x, nfft=nfft, fs=fs, window=window, scaling='density', detrend=False, noverlap=noverlap)

        # Calculate slope of PSD
        slo = np.gradient(p)

        # Get the selected parts for the band, the noise and the total noise
        lcf20_f, lcf20_p = select_fin_band(p, f, 'LFC20')
        lcf20_f_noise, lcf20_p_noise = select_noise_band(p, f, 'LFC20')
        lcf20_f_noise_total, lcf20_p_noise_total = select_total_noise_band(p, f, 'LFC20')

        hcf80_f, hcf80_p = select_fin_band(p, f, 'HFC80')
        hcf80_f_noise, hcf80_p_noise = select_noise_band(p, f, 'HFC80')
        hcf80_f_noise_total, hcf80_p_noise_total = select_total_noise_band(p, f, 'HFC80')

        hcf90_f, hcf90_p = select_fin_band(p, f, 'HFC90')
        hcf90_f_noise, hcf90_p_noise = select_noise_band(p, f, 'HFC90')
        hcf90_f_noise_total, hcf90_p_noise_total = select_total_noise_band(p, f, 'HFC90')

        # Extract average PSDs of chorus bands
        fl_1 = np.mean(lcf20_p)
        fl_2 = np.mean(hcf80_p)
        fl_3 = np.mean(hcf90_p)

        # Extract median PSDs of noise bands
        n_1 = np.median(lcf20_p_noise)
        n_2 = np.median(hcf80_p_noise)
        n_3 = np.median(hcf90_p_noise)

        # Extract area under PSD curve within chorus bands and noise bands
        a1 = np.trapz(lcf20_p, lcf20_f) / np.trapz(lcf20_p_noise_total, lcf20_f_noise_total)
        a2 = np.trapz(hcf80_p, hcf80_f) / np.trapz(hcf80_p_noise_total, hcf80_f_noise_total)
        a3 = np.trapz(hcf90_p, hcf90_f) / np.trapz(hcf90_p_noise_total, hcf90_f_noise_total)

        # Extract slopes at borders of chorus
        s1 = np.max(slo[(f >= (LFC20['f0'] - 2)) & (f <= (LFC20['f0'] + 2))])
        s2 = np.max(slo[(f >= (HFC80['f0'] - 2)) &
                        (f <= (HFC80['f0'] + 2))]) - np.min(slo[(f >= (HFC80['f1'] - 2)) & (f <= (HFC80['f1'] + 2))])
        s3 = np.max(slo[(f >= (HFC90['f0'] - 2)) &
                        (f <= (HFC90['f0'] + 2))]) - np.min(slo[(f >= (HFC90['f1'] - 2)) & (f <= (HFC90['f1'] + 2))])

        # Compute SNRs and store all values in the prepared table
        snr_20 = (10*np.log10(fl_1)) - (10*np.log10(n_1))
        snr_80 = (10*np.log10(fl_2)) - (10*np.log10(n_2))
        snr_90 = (10*np.log10(fl_3)) - (10*np.log10(n_3))
        parameters.loc[len(parameters), :] = [str(file_path), snr_20, snr_80, snr_90, a1, a2, a3, s1, s2, s3]

    # Filter table for thresholds
    selected_parameters = parameters.loc[(parameters.SNRL >= TH[0]) & (parameters.SNR8 >= TH[1]) &
                                         (parameters.SNR9 >= TH[2]) & (parameters.AL >= TH[3]) &
                                         (parameters.A8 >= TH[4]) & (parameters.A9 >= TH[5]) &
                                         (parameters.SL >= TH[6]) & (parameters.S8 >= TH[7]) & (parameters.S9 >= TH[8])]

    # save table with computed values - edit path and filename to execute
    parameters.to_csv(save_path.joinpath('chorus_parameters.csv'))
    selected_parameters.to_csv(save_path.joinpath('detected_choruses.csv'))


def plot_noise_distribution(survey_path, save_path):
    """
    :param survey_path: path to folder (can have subfolders) wich contains your audio data - edit to execute
    :return:
    """
    # Make sure the survey path is in the correct format
    if not isinstance(survey_path, pathlib.Path):
        survey_path = pathlib.Path(survey_path)
    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)

    total_wavs = len(list(survey_path.glob('*.wav')))
    # Loop through wav files to compute metrics
    total_psd = np.zeros((total_wavs, 1025))
    for file_i, file_path in tqdm(enumerate(survey_path.glob('*.wav')), total=total_wavs):
        # Get sampling rate and duration info from wave file
        sound_file = sf.SoundFile(file_path)
        fs = sound_file.samplerate

        # Read and normalize wave file
        x = sound_file.read()
        x = stats.zscore(x)

        # If the sampling rate is not 500, downsample
        if fs > 500:
            lcm = np.lcm(int(fs), int(500))
            ratio_up = int(lcm / fs)
            ratio_down = int(lcm / 500)
            lowpass_filt = sig.butter(4, 500/2, btype='low', fs=fs, output='sos')
            x = sig.sosfilt(lowpass_filt, x)
            x = sig.resample_poly(x, ratio_up, ratio_down)
            fs = 500

        # Calculate PSD (p) of wav file and extract frequency resolution (f)
        nfft = 2048
        noverlap = int(nfft/2)
        window = sig.windows.hamming(nfft)
        f, p = sig.welch(x, nfft=nfft, fs=fs, window=window, scaling='density', detrend=False, noverlap=noverlap)
        total_psd[file_i, :] = p

    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    mu, std = norm.fit(total_psd.flatten())
    # histogram
    counts, bins = np.histogram(total_psd.flatten(), bins='auto', range=(0, 0.04), density=True)
    # plot the histogram
    plt.stairs(counts, bins)
    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(survey_path.parent.name)
    plt.show()


if __name__ == "__main__()":
    # Provide path to folder with wave files to process - edit path to execute
    path = pathlib.Path('/Users/../Documents/.../*.wav')  # Get file info from all wav files in folder
    detect_chorus(path)
