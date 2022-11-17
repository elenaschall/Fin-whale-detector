import numpy as np
import pathlib
import soundfile as sf
import pandas as pd
import scipy.signal as sig
import datetime
from scipy.stats import kurtosis
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *

# Thresholds from the paper
TH = {
    'tk1': 2.5,
    'tk2': 40,
    'tsf': 8,
    'tst': -2,
    'tst2': -8,
    'tbw': 75,
    'tk1_2': 4.75
}


def detect_f20p(survey_path, save_path):
    if not isinstance(survey_path, pathlib.Path):
        survey_path = pathlib.Path(survey_path)
    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)
    # Compute the parameters needed to make the detections
    Tkurt = compute_params(survey_path)
    Tkurt.to_csv(save_path.joinpath('f20p_parameters.csv'))
    # Tkurt = pd.read_csv(save_path.joinpath('f20p_parameters.csv'), parse_dates=['DateTime'], index_col=0)

    # Join all the detections within 2.5 seconds
    Tkurt = join_close_detections(Tkurt)

    # Filter table for thresholds
    Tkurt_f = apply_thresholds(Tkurt, **TH)

    # Filter to only preserve clusters of minimum 5 detections within 5 minutes
    Tkurt_f = filter_lonely_detections(Tkurt_f)

    # save table with computed values - edit path and filename to execute
    Tkurt_f.to_csv(save_path.joinpath('detected_f20p.csv'))


def compute_params(survey_path):
    """
    :param survey_path: path to folder (can have subfolders) which contains your audio data - edit to execute
    :return:
    """
    # Create empty table to store values and set counter to 0 to write into table
    Tkurt = pd.DataFrame()
    c = 0

    # Loop through wave files to compute metrics
    # Get file info from all wav files in folder and subfolders
    for x_path in tqdm(survey_path.glob('*.wav'), total=len(list(survey_path.glob('*.wav')))):
        # Get sampling rate and duration info from wave file and store duration in list
        sound_file = sf.SoundFile(x_path)
        fs = sound_file.samplerate
        dur = sound_file.frames / fs
        # survey_list(x).Dur = dur

        # Retrieve date and time info from filename to add to detection time
        dt_name = x_path.name[0:15]
        dt_format = 'yyyy-MM-dd HH:mm:ss.SSSS'
        try:
            file_time = datetime.datetime.strptime(dt_name, '%Y%m%d-%H%M%S')
        except ValueError:
            file_time = datetime.datetime.strptime(dt_name, '%Y%m%d_%H%M%S')

        # Create bandpass filter according to sampling rate of file
        bp_filter = sig.iirfilter(N=10, Wn=[F20P['f0'], F20P['f1']], btype='bandpass', ftype='butter', analog=False, fs=fs,
                               output='sos')

        # Read audiofile
        y = sound_file.read()

        # Process audiofiles in 2s-sliding windows with 1.5s overlap
        for i in np.arange(start=0.5, step=0.5, stop=dur - 2):
            # cut audio snippet
            start_sample = int(i * fs) - 1
            end_sample = int((i + 2) * fs) - 1
            y_t = y[start_sample: end_sample]

            # compute PSD of snippet
            window = sig.windows.hamming(fs)
            noverlap = int(fs/2)
            f, p = sig.welch(y_t, window=window, nfft=fs, fs=fs, noverlap=noverlap, detrend=False, scaling='density')

            # Bandpass filter snippet
            y_f = sig.sosfilt(bp_filter, y_t)

            # Apply Teager-Kaiser-Energy Operator (TKEO)
            ey, ex = energyop(y_f)

            # Retreive sample index of maximum TKEO value for timestamp of detection
            idx = ex.argmax()
            m = ex[idx]

            # Update counter and write filename and time of detection into table
            c = c + 1
            Tkurt.loc[c, 'File'] = x_path
            Tkurt.loc[c, 'DateTime'] = file_time + datetime.timedelta(seconds=i + (idx / fs))

            # Store kurtosis value and kurtosis product oin table
            Tkurt.loc[c, 'Kurt'] = kurtosis(y_f)
            Tkurt.loc[c, 'KurtProd'] = np.multiply(kurtosis(y_f), kurtosis(ex))

            # Calculate spectral SNR with frequency band limits and store into table
            _, f20p_p = select_fin_band(p, f, 'LFC20')
            _, f20p_p_noise = select_noise_band(p, f, 'LFC20')

            Tkurt.loc[c, 'SNRF'] = 10 * np.log10(np.mean(f20p_p)) - 10 * np.log10(np.mean(f20p_p_noise))

            # Split audio snippet into signal of 0.6s and noise before and after
            # (0.6s) depending on position of snippet within audio file
            if i <= 1.5:
                y_post = y[int(np.ceil((i + 2) * fs - 1)): int(np.ceil((i + 2) * fs - 1 + 2 * fs)) + 1]
                y_t = np.concatenate([y_t, y_post])
                if idx - np.round(0.6 * fs) <= 0:
                    signal = y_t[0: int(idx + np.round(0.6 * fs))]
                else:
                    signal = y_t[int(idx - np.round(0.6 * fs)): int(idx + np.round(0.6 * fs))]

                noise = y_t[int(idx + np.round(0.6 * fs)): int(idx + np.round(1.2 * fs))]
            elif i >= dur - 3:
                y_pre = y[int(np.floor(i * fs - 2 * fs)): int(np.floor(i * fs))]
                y_t = np.concatenate([y_pre, y_t])
                idx = len(y_pre) + idx
                if idx + np.round(0.6 * fs) > len(y_f):
                    signal = y_t[int(idx - np.round(0.6 * fs)): -1]
                else:
                    signal = y_t[int(idx - np.round(0.6 * fs)): int(idx + np.round(0.6 * fs))]

                noise = y_t[int(idx - np.round(1.2 * fs)): int(idx - np.round(0.6 * fs))]
            else:
                y_pre = y[int(np.ceil(i * fs - np.round(1.2 * fs))): int(np.ceil(i * fs))]
                y_post = y[int(np.floor((i + 2) * (fs - 1))): int(np.floor((i + 2) * (fs - 1) + np.round(1.2 * fs)))]
                y_t = np.concatenate([y_pre, y_t, y_post])
                idx = len(y_pre) + idx
                signal = y_t[int(idx - np.round(0.6 * fs)): int(idx + np.round(0.6 * fs))]
                noise_pre = y_t[int(idx - np.round(1.2 * fs)): int(idx - np.round(0.6 * fs))]
                noise_post = y_t[int(idx + np.round(0.6 * fs)): int(idx + np.round(1.2 * fs))]
                noise = np.concatenate([noise_pre, noise_post])

            # Calculate temporal SNR and store into table
            filtered_sig = sig.sosfilt(bp_filter, signal)
            filtered_noise = sig.sosfilt(bp_filter, noise)
            Tkurt.loc[c, 'SNRT'] = 20 * np.log10(abs(rms(filtered_sig) - rms(filtered_noise)) / rms(filtered_noise))

            # Compute PSD of signal and noise and calculate the SNRs for bandwidth
            f, p = sig.welch(signal, nfft=fs, fs=fs, scaling='density', detrend=False)
            ps = p[(f >= 13) & (f <= 35)]
            f, p = sig.welch(noise, nfft=fs, fs=fs, scaling='density', detrend=False)
            pn = p[(f >= 13) & (f <= 35)]
            BWSNR = 10 * np.log10(ps) - 10 * np.log10(pn)

            # Store BW in table by counting the number of frequency bins where
            # SNR is above 0
            Tkurt.loc[c, 'BW'] = sum(BWSNR >= 0)

    return Tkurt


def join_close_detections(Tkurt):
    # Join detections within 2s range by keeping maximum kurtosis, earliest time, and average SNRs
    buff_dt = datetime.timedelta(seconds=2)
    new_tkurt = pd.DataFrame(columns=Tkurt.columns)
    for i in Tkurt.index:
        mask = Tkurt.loc[
            ((Tkurt.DateTime >= (Tkurt.loc[i].DateTime - buff_dt)) & (Tkurt.DateTime <= (Tkurt.loc[i].DateTime + buff_dt)))]
        if len(mask) == 1:
            new_tkurt = pd.concat([new_tkurt, mask])
        else:
            first_row = mask.iloc[0].copy()
            first_row.Kurt = Tkurt.Kurt.loc[mask.index].max()
            first_row.KurtProd = Tkurt.KurtProd[mask.index].max()
            first_row.SNRF = Tkurt.SNRF[mask.index].max()
            first_row.SNRT = Tkurt.SNRT[mask.index].max()
            first_row.BW = Tkurt.BW[mask.index].max()
            new_tkurt.loc[first_row.name] = first_row

    return new_tkurt


def apply_thresholds(df, tk1, tk2, tsf, tst, tst2, tbw, tk1_2):
    selected_rows = df.loc[(df.Kurt >= tk1) | (df.KurtProd >= tk2)]
    selected_rows = selected_rows.loc[selected_rows.SNRF >= tsf]
    selected_rows = selected_rows.loc[(selected_rows.SNRT >= tst2) | ((selected_rows.SNRT >= tst)
                                                                      & (selected_rows.Kurt >= tk1_2)
                                                                      & (selected_rows.BW > tbw))]

    # edit values in TH vector to filter for specific threshold values in the

    # Tkurt_f = Tkurt.loc[(Tkurt.Kurt >= TH[0]) | (Tkurt.KurtProd >= TH(1))]
    # Tkurt_f = Tkurt_f.loc[Tkurt_f.SNRF >= TH(2)]
    # Tkurt_f = Tkurt_f.loc[(Tkurt_f.SNRT >= TH(3)) | (Tkurt_f.SNRT >= TH(4))
    #                       & (Tkurt_f.BW >= (23 / 100 * TH(5))) & (Tkurt_f.Kurt >= TH(6))]
    return selected_rows


def apply_thresholds_probab(df, tk1, tk2, tsf, tst, tst2, tbw, tk1_2, p0, p1, p2, p3, p4, p5):
    df['probab'] = -1
    sr = df.loc[(df.Kurt >= tk1) | (df.KurtProd >= tk2)]
    df.loc[~df.index.isin(sr.index), 'probab'] = p0
    sr = sr.loc[sr.SNRF >= tsf]
    df.loc[~df.index.isin(sr.index), 'probab'] = p1
    sr = sr.loc[(sr.SNRT >= tst)]
    df.loc[~df.index.isin(sr.index), 'probab'] = p2
    fin_0 = sr.loc[(sr.SNRT >= tst2)]
    sr = sr.loc[~sr.index.isin(fin_0)]
    df.loc[df.index.isin(fin_0.index), 'probab'] = p3
    sr = sr.loc[(sr.BW >= tbw)]
    df.loc[~df.index.isin(sr.index), 'probab'] = p4
    last_noise = sr.loc[~(sr.Kurt >= tk1_2)]
    sr = sr.loc[sr.Kurt >= tk1_2]
    df.loc[df.index.isin(last_noise.index), 'probab'] = 1 - p5
    df.loc[df.index.isin(sr.index), 'probab'] = p5

    return sr, df


def filter_lonely_detections(positive_detections):
    dt = datetime.timedelta(minutes=2.5)
    positive_detections['corrected_prediction'] = 1
    for i, row in positive_detections.iterrows():
        detections_surroundings = ((positive_detections.DateTime >= (row.DateTime - dt))
                                       & (positive_detections.DateTime <= (row.DateTime + dt))).sum()
        if detections_surroundings < 5:
            positive_detections.loc[i, 'corrected_prediction'] = 0

    # qs = []
    # buff_dt_min = datetime.timedelta(seconds=2.5 * 60)
    # for q in np.arange(len(Tkurt_f)):
    #     if q > len(Tkurt_f):
    #         break
    #     c5 = len(Tkurt_f.loc[Tkurt_f.DateTime >= (Tkurt_f.loc[q, 'DateTime'] - buff_dt_min)
    #                          & (Tkurt_f.DateTime <= Tkurt_f.loc[q, 'DateTime'] + buff_dt_min)])
    #     if c5 < 5:
    #         qs.append(q)
    #
    # Tkurt_f = Tkurt_f.loc[~Tkurt_f.index.isin(qs)]

    return positive_detections.loc[positive_detections['corrected_prediction'] == 1]


# Define the function to predict
def make_one_prediction(row, tk1, tk2, tsf, tst, tst2, tbw, tk1_2):
    if (row.KurtProd >= tk2) or (row.Kurt >= tk1):
        if row.SNRF >= tsf:
            # Should be the same than
            # (row.SNRT >= tst) or (row.SNRT >= tst2) and (row.BW >= tbw) and (row.Kurt >= tk1_2)
            if row.SNRT >= tst2:
                if row.SNRT >= tst:
                    return 1
                else:
                    if row.BW >= tbw:
                        if row.Kurt >= tk1_2:
                            return 1
                        else:
                            return 0
                    else:
                        return 0
            else:
                return 0
        else:
            return 0
    else:
        return 0
