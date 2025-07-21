# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:14:59 2024

@author: jmarti2
"""

import os.path as op
import os
import pathlib as pl
import glob
import shutil
import re

import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.signal import medfilt
import numpy as np
import seaborn as sns


subjects = ['3035']


def moving_window_3sd_with_times(data, times, window_size):
    # Convert the input data and times to numpy arrays if they are not already
    data = np.array(data, dtype=float)
    times = np.array(times, dtype=float)
    
    # Create an output array with the same shape as data, initialized to be a copy of data
    output_data = np.copy(data)
    output_times = np.copy(times)
    
    # Lists to store removed data and times
    removed_data = []
    removed_times = []
    
    # Iterate over the array with a moving window
    for i in range(window_size, len(data) - window_size):
        # Define the window of data points around the current index
        window = data[i - window_size:i + window_size + 1]
        
        # Calculate the mean and standard deviation of the window
        window_mean = np.mean(window)
        window_std = np.std(window)
        
        # If the data point exceeds 3 standard deviations, set it to NaN in both arrays
        if abs(data[i] - window_mean) > 3 * window_std:
            # Add the removed data and times to the lists
            removed_data.append(data[i])
            removed_times.append(times[i])
            
            # Set the data and time at this index to NaN
            output_data[i] = np.nan
            output_times[i] = np.nan
    
    return output_data, output_times, removed_data, removed_times


def detect_spikes_first_derivative(data, sample_rate=50, threshold=None, percentile=95):
    """
    Detect spikes in time-series data using the first derivative and return interpolated data.

    Parameters:
    - data: array-like
        Time-series data to analyze.
    - sample_rate: int, optional
        Sampling rate of the data in Hz (default: 50 Hz).
    - threshold: float, optional
        Velocity threshold to detect spikes. If None, dynamic threshold using percentile is applied.
    - percentile: float, optional
        Percentile for dynamic threshold if `threshold` is None (default: 95).

    Returns:
    - interpolated_data: np.ndarray
        Data with spikes interpolated.
    - spike_indices: np.ndarray
        Indices where spikes were detected.
    - velocity: np.ndarray
        First derivative (velocity) of the dataw.
    """
    # Compute velocity
    velocity = np.diff(data) * sample_rate

    # Determine threshold dynamically if not provided
    if threshold is None:
        threshold = np.percentile(np.abs(velocity), percentile)

    # Detect spike indices
    spike_indices = np.where(np.abs(velocity) > threshold)[0]

    # Interpolate spike points
    interpolated_data = np.copy(data)
    for idx in spike_indices:
        if 1 < idx < len(interpolated_data) - 1:
            interpolated_data[idx] = (interpolated_data[idx - 1] + interpolated_data[idx + 1]) / 2

    return interpolated_data, spike_indices, velocity


def velocity_filter(pupil_data, sample_rate=50, threshold=None, percentile=95):
    """
    Apply a velocity-based filter to clean pupillometry data by detecting and interpolating spikes.

    Parameters:
    - pupil_data: array-like
        Pupil size data (e.g., in mm).
    - sample_rate: int, optional
        Sampling rate of the data in Hz (default: 50 Hz).
    - threshold: float, optional
        Maximum allowable velocity (in units per second). If None, a dynamic threshold is used.
    - percentile: float, optional
        Percentile to compute dynamic velocity threshold if `threshold` is None (default: 95).

    Returns:
    - corrected_data: np.ndarray
        Filtered pupil data with spikes removed.
    - velocity: np.ndarray
        The computed velocity values.
    """
    # Compute velocity (first derivative scaled by sample rate)
    velocity = np.diff(pupil_data) * sample_rate

    # Determine threshold dynamically if not provided
    if threshold is None:
        threshold = np.percentile(np.abs(velocity), percentile)
        
    # Detect spike indices where the velocity exceeds the threshold
    spike_indices = np.where(np.abs(velocity) > threshold)[0]

    # Create a copy of the data for correction
    corrected_data = np.copy(pupil_data)

    # Interpolate spike points
    for idx in spike_indices:
        if 1 < idx < len(corrected_data) - 1:
            corrected_data[idx] = (corrected_data[idx - 1] + corrected_data[idx + 1]) / 2

    return corrected_data, velocity


def moving_average_filter(data, window_size=5):
    """
    Apply a simple moving average filter to the data.

    Parameters:
    - data: array-like
        Input time-series data.
    - window_size: int, optional
        Size of the moving average window (default: 5).

    Returns:
    - filtered_data: np.ndarray
        Smoothed data after applying the moving average filter.
    """
    if window_size < 1:
        raise ValueError("Window size must be a positive integer.")

    # Apply moving average filter using convolution
    kernel = np.ones(window_size) / window_size
    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data


def diff_spike_filter(data, diff_threshold=3):
    # Compute first-order differences
    data_diff = np.abs(np.diff(data, prepend=data[0]))  # prepend to match length
    
    # Threshold based on standard deviation of the differences
    threshold = diff_threshold * np.nanstd(data_diff)
    
    # Identify spikes where differences exceed the threshold
    spikes = data_diff > threshold
    
    # Mask spikes by setting them to NaN
    filtered_data = np.where(spikes, np.nan, data)
    return filtered_data


def mad_outlier_filter(data, threshold=3):
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    mask = np.abs(data - median) > threshold * mad
    filtered_data = np.where(mask, np.nan, data)
    return filtered_data


def gradient_spike_filter(data, threshold=0.1):
    gradient = np.abs(np.diff(data))
    spikes = gradient > threshold * np.nanmax(gradient)
    filtered_data = np.array(data)
    filtered_data[1:][spikes] = np.nan  # Ignore first element
    return filtered_data


def threshold_outlier_filter(data, threshold=3):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    mask = np.abs(data - mean) > threshold * std
    filtered_data = np.where(mask, np.nan, data)
    return filtered_data

def detect_interpolated_segments(pupil_data, window_size=20, r_squared_threshold=.9999):
    """
    Detects likely interpolated segments in pupil size data based on linear patterns.
    
    Parameters:
        pupil_data (np.ndarray): 1D array of pupil size data.
        window_size (int): Number of points in each rolling window to check for linearity.
        r_squared_threshold (float): Minimum R-squared value to classify a segment as interpolated.
    
    Returns:
        np.ndarray: Boolean array where True indicates likely interpolated samples.
    """
    interpolated_flags = np.zeros(len(pupil_data), dtype=bool)

    # Loop over each window in the data
    for i in range(len(pupil_data) - window_size + 1):
        window = pupil_data[i:i + window_size]

        # Perform linear regression on the window
        slope, intercept, r_value, p_value, std_err = linregress(
            range(window_size), window)

        # Check if R-squared value meets threshold for linearity
        if r_value**2 >= r_squared_threshold:
            interpolated_flags[i:i + window_size] = True

    return interpolated_flags


# Path to data
datastore = pl.Path("Z:\Part B\Pupil_data")

# Get the files
files = list(datastore.rglob("*/Experiment_0001/data*.csv"))

if isinstance(subjects, list):
    compiled_patterns = [re.compile(fr'_{pattern}\\') for pattern in subjects]
    files = [
    file for file in files
    if any(pattern.search(str(file)) for pattern in compiled_patterns)
]
elif subjects == 'all':
    print('Analysing data for all subjects')
    
# Loop over the files
for fpath in files:
    # Pull out some info from the filename
    subject = re.findall(r'_(\d{4})\\E', str(fpath))[0]
    age = re.findall(r'_(\d{2})_', str(fpath))[0]
    protocol = re.findall(r'HELIOS_([A-Z]+)', str(fpath))[0]
    
    print('*******************')
    print(f'Subject: {subject}')
    print(f'Protocol: {protocol}')
    print(f'Data: {fpath.stem}')
    print('*******************')

    # Get subject directory and create output directory
    subject_dir = fpath.parent.parent
    out_dir_name = 'data_smooth' if fpath.stem == 'data' else fpath.stem
    out_dir = subject_dir / f'{out_dir_name}_out'
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)    

    # temp
    temp_out_dir = subject_dir / 'out'
    if temp_out_dir.exists():
        shutil.rmtree(temp_out_dir)    
    
    # Load pupil data
    df = pd.read_csv(fpath, sep=";")
    dfr = df.loc[df["Right - Is found"] == True]
    dfl = df.loc[df["Left - Is found"] == True]

    # Plot the raw traces
    fig, ax = plt.subplots(figsize=(12, 4))
    dfr["Right - Size Mm"].plot(ax=ax, label="Right eye")
    dfl["Left - Size Mm"].plot(ax=ax, label="Left eye")
    ax.set(xlabel="Time (s)", ylabel="Pupil size (mm)",
           title=f'{subject}: {protocol.upper()}')
    ax.legend()
    fig.savefig(out_dir / f"raw_{protocol}_pupil_traces.png")
    plt.show()

    # Set params for data type
    if protocol == "LMS":
        sequence_indices = range(2, 17)
        color = 'tab:green'
    elif protocol == "MELANOPSIN":
        sequence_indices = range(2, 17)
        color = 'tab:blue'
    elif protocol == "PIPR":
        sequence_indices = range(1, 13)
        palette = {'red': 'tab:red', 'blue': 'tab:blue'}
    else:
        raise ValueError(
            f"protocol must be 'pipr', 'lms', or 'mel' (not '{protocol}')"
        )

    # Specify new times for pupil data
    newt = np.linspace(-1, 17.0, 18 * 50)
    dfs = []
    trial = 0
    for si in sequence_indices:
        # Get the current sequence
        sequence_data = dfr.loc[dfr["Sequence index"] == si]
        # Get the label from original df
        try:
            # Get the condition label
            label = df.loc[
                (
                    (df["Sequence index"] == si)
                    & (
                        df["Excitation label - Right"].isin(
                            ["red", "blue", "lms", "mel"]
                        )
                    )
                ),
                "Excitation label - Right",
            ].iloc[0]
            
            # Get the baseline
            baseline = sequence_data.loc[
                sequence_data["Excitation label - Right"] == "baseline", "Right - Size Mm"
            ].mean()
            
            # Get the times
            times = dfr.loc[dfr["Sequence index"] == si, "Sequence time Sec"]
            
            # Get pupil size mm and pupil size percent change
            pupil = dfr.loc[dfr["Sequence index"] == si, "Right - Size Mm"]
            
            # Use a 100 sample sliding window and a 3SD filter to detect outliers.
            # These are spurious samples caused by imperfect pupil tracking.
            pupil, times, removed_pupil, removed_times = moving_window_3sd_with_times(pupil, times, window_size=100)
            
            base_corrected_pupil = (
                dfr.loc[dfr["Sequence index"] == si,
                        "Right - Size Mm"] / baseline * 100
            )
                  
            # Get the first time point of the relevant excitation index
            on = df.loc[
                (
                    (df["Sequence index"] == si)
                    & (df["Excitation label - Right"] == label)
                )
            ].iloc[0]["Sequence time Sec"]
            times = times - on
            
            # Interpolate to new times
            fp = scp.interpolate.interp1d(times, pupil, fill_value="extrapolate")
            fbp = scp.interpolate.interp1d(
                times, base_corrected_pupil, fill_value="extrapolate")
            newp = fp(newt)
            newbp = fbp(newt)

            # Filter, cutoff is 4/(sample_rate/2)
            B, A = scp.signal.butter(3, 4 / (50 / 2), output="BA")
            filt_newp = scp.signal.filtfilt(B, A, newp)
            filt_newbp = scp.signal.filtfilt(B, A, newbp)
            
            #sg_newp = savgol_filter(newp, window_length=51, polyorder=3)
            #sg_newbp = savgol_filter(newbp, window_length=51, polyorder=3)

            # Figure for the trial
            fig, ax = plt.subplots()
            ax.scatter(times, pupil, color='k', alpha=.4, s=8, label='raw samples')
            ax.scatter(removed_times, removed_pupil,marker='x', color='purple', s=8, label='removed raw samples')
            
            # Find which segments were interpolated and plot the trial
            if protocol == 'PIPR':
                color = f'tab:{label}'
            interpolated = detect_interpolated_segments(newp)
            ax.plot(newt, newp, lw=2, ls=':', c=color, label='interpolated')
            ax.plot(newt, filt_newp, lw=1, c=color, label='filtered')


            ylims = ax.get_ylim()
            ax.fill_between(newt, ylims[0], ylims[1], where=interpolated,
                             color='red', alpha=0.3, label='Suspicious data')
            ax.fill_between(
                (0, 3), ylims[0], ylims[1], alpha=0.2, color="k", label='Stimulus'
            )
            ax.set(
                title=f'{subject}_{trial}_{protocol}',
                xlabel='Time (s)',
                ylabel='Pupil size (mm)'
                       )
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            fig.savefig(out_dir / f"trial_{trial}_{protocol}.png", bbox_inches='tight')
            plt.show()            

            # Create new dataframe
            newdf = pd.DataFrame(
                data={
                    "pct_interpolated": interpolated.sum() / len(interpolated),
                    "interpolated": interpolated,
                    "pc_pupil": newbp,
                    "pupil": newp,
                    "filt_pc_pupil": filt_newbp,
                    "filt_pupil": filt_newp,
                    "condition": label,
                    "time": newt,
                    "trial": trial,
                    "subject": subject,
                    "baseline": baseline,
                    'age': age
                }
            )
            dfs.append(newdf)
        except:
            continue
        trial += 1
        
    # Save DF
    gdf = pd.concat(dfs).reset_index(drop=True)
    gdf.to_csv(out_dir / f"processed_{protocol}_PLRs_{subject}.csv")
    
    gdf = gdf.query("(baseline.notna()) and (pct_interpolated < .3)")

    # Make plots
    fig, ax = plt.subplots()
    if protocol in ['LMS', 'MELANOPSIN']:
        sns.lineplot(
            data=gdf,
            x="time",
            y="pc_pupil",
            color=color,
            ax=ax,
        )
    elif protocol=='PIPR':
        sns.lineplot(
            data=gdf,
            x="time",
            y="pc_pupil",
            hue='condition',
            palette=palette,
            ax=ax,
        )
    ax.set(xlabel="Time (s)", ylabel="Pupil size (%-change)", title=f'{subject}: {protocol.upper()}')
    ax.fill_between(
            (0, 3), min(ax.get_ylim()), max(ax.get_ylim()), alpha=0.2, color="k"
        )
    ax.grid()
    ax.legend(loc="lower right")
    fig.savefig(out_dir / f"processed_{protocol}_PLR_ave.png")

    # Plot raw traces
    fig, ax = plt.subplots()
    if protocol in ['LMS', 'MELANOPSIN']:
        # Plot raw traces
        sns.lineplot(
            data=gdf,
            x="time",
            y="pc_pupil",
            color=color,
            ax=ax,
            units='trial',
            estimator=None
        )
    elif protocol=='PIPR':
        sns.lineplot(
            data=gdf,
            x="time",
            y="pc_pupil",
            hue='condition',
            palette=palette,
            ax=ax,
            units='trial',
            estimator=None
        )
    ax.set(xlabel="Time (s)", ylabel="Pupil size (%-change)", title=f'{subject}: {protocol.upper()}')
    ax.fill_between(
        (0, 3), min(ax.get_ylim()), max(ax.get_ylim()), alpha=0.2, color="k"
    )
    ax.grid()
    ax.legend(loc="lower right")
    fig.savefig(out_dir / f"processed_{protocol}_PLR_traces.png")
