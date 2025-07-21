# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:14:59 2024

@author: jmarti2
"""

import pathlib as pl
import shutil
import re

import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
import numpy as np
import seaborn as sns

# Enter the subjects that you want to analyse. This should be either a list of
# numerical IDs, or simply 'all'
#subjects = 'all'  # Will process all data
subjects = ['999']  # Will process all runs for participant 999

# Path to data. Set this to the correct location.
datastore = pl.Path(r"C:\Users\jmarti2\OneDrive - University of Edinburgh\AMBIENT-BD\Results")

##########################################################################
# DON'T CHANGE ANYTHING BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING :) #
##########################################################################

# Get the files
files = list(datastore.rglob("*/Experiment_0001/data_pupil_od_lateral.csv"))

if isinstance(subjects, list):
    compiled_patterns = [re.compile(fr'ABD{pattern}') for pattern in subjects]
    files = [
    file for file in files
    if any(pattern.search(str(file)) for pattern in compiled_patterns)
]
elif subjects == 'all':
    print('Analysing data for all subjects')
    
# Loop over the files
for fpath in files:
    # Pull out some info from the filename
    subject = re.findall(r'ABD(\d{3})', str(fpath))[0]
    age = re.findall(r'_(\d{2})_', str(fpath))[0]
    protocol = re.findall(r'AMBIENT_([A-Z]+)', str(fpath))[0]
    run = re.findall(r'\d{3}_(\d{1})', str(fpath))[0]
    
    print('*******************')
    print(f'Subject: {subject}')
    print(f'Protocol: {protocol}')
    print(f'Data: {fpath.stem}')
    print('*******************')

    # Get subject directory and create output directory
    subject_dir = fpath.parent.parent
    out_dir_name = 'data' if fpath.stem == 'data' else fpath.stem
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
    df['Trial_id'] = df['Excitation label'].str.split('-').str[0]
    df['Trial_id'] = df['Trial_id'].astype(int)
    df['Stimulus'] = df['Excitation label'].str.split('-').str[1]
    df['Status'] = df['Excitation label'].str.split('-').str[2]

    # Plot the raw traces
    fig, ax = plt.subplots(figsize=(12, 4))
    df["Pupil size, mm"].plot(ax=ax, label="Right eye")
    ax.set(xlabel="Time (s)", ylabel="Pupil size (mm)",
           title=f'{subject}: {protocol.upper()}')
    ax.legend()
    fig.savefig(out_dir / f"raw_{protocol}_pupil_traces.png")
    plt.show()

    # Set params for data type
    sequence_indices = range(1, 13)
    palette = {'red': 'tab:red', 'blue': 'tab:blue'}


    # Specify new times for pupil data
    newt = np.linspace(-1, 17.0, 18 * 50)
    dfs = []
    
    # Incrementor variable for trial
    trial = 0
    
    for si in sequence_indices:
        # Get the current sequence
        sequence_data = df.loc[df["Trial_id"] == si]
        try:
            # Get the condition label
            label = sequence_data['Stimulus'].iloc[0]
            
            # Get the trial id
            trial_id = sequence_data['Trial_id'].iloc[0]
            
            # Get the baseline
            baseline = sequence_data.loc[
                sequence_data["Status"] == "baseline", "Pupil size, mm"
            ].mean()
            
            # Get the times
            times = sequence_data['Experiment time, sec']
            
            # Get pupil size mm 
            pupil = sequence_data['Pupil size, mm']
            
            # Sometimes the neighbouring values have sharp drops due to rapid
            # eyelid occlusion so we will broaden the nan periods. If any 
            # sample has a neighboring value that is nan, we will make it also
            # a nan
            neighbor_nan = pupil.shift(1).isna() | pupil.shift(-1).isna()
            pupil[neighbor_nan] = np.nan
            
            # Calculate baseline corrected pupil size
            base_corrected_pupil = (
                pupil / baseline * 100
            )
                  
            # Get the first time point of the relevant excitation index
            on = sequence_data.loc[sequence_data['Status']=='stimulation', 'Experiment time, sec'].iloc[0]

            # Subtract from all time points
            times = times - on
            
            # Interpolate to new times
            fp = scp.interpolate.interp1d(times, pupil, fill_value="extrapolate")
            fbp = scp.interpolate.interp1d(
                times, base_corrected_pupil, fill_value="extrapolate")
            newp = fp(newt)
            newbp = fbp(newt)
            
            # Keep a record of the missing data
            interpolated = np.isnan(newp)
            
            # Now use linear interpolation. Do it once, reverse, then do it 
            # again. This captures any NaNs at the start of the sequence.
            newp = pd.Series(newp).interpolate()[::-1].interpolate()[::-1]
            newbp = pd.Series(newbp).interpolate()[::-1].interpolate()[::-1]

            # Filter, cutoff is 4/(sample_rate/2)
            B, A = scp.signal.butter(3, 4 / (50 / 2), output="BA")
            filt_newp = scp.signal.filtfilt(B, A, newp)
            filt_newbp = scp.signal.filtfilt(B, A, newbp)
            
            # Figure for the trial
            fig, ax = plt.subplots()
            ax.scatter(times, pupil, color='k', alpha=.4, s=10, label='raw samples')
            
            # Find which segments were interpolated and plot the trial
            if protocol == 'PIPR':
                color = f'tab:{label}'
            ax.plot(newt, newp, lw=2, ls=':', c=color, label='interpolated')
            ax.plot(newt, filt_newp, lw=1, c=color, label='filtered')


            ylims = ax.get_ylim()
            ax.fill_between(newt, ylims[0], ylims[1], where=interpolated,
                             color='red', alpha=0.3, label='NaN')
            ax.fill_between(
                (0, 3), ylims[0], ylims[1], alpha=0.2, color="k", label='Stimulus on'
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
                    "trial_num": trial,
                    "trial_id": trial_id,
                    "subject": subject,
                    "baseline": baseline,
                    'age': age,
                    'run': run
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
    sns.lineplot(
        data=gdf,
        x="time",
        y="filt_pc_pupil",
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
    sns.lineplot(
        data=gdf,
        x="time",
        y="filt_pc_pupil",
        hue='condition',
        palette=palette,
        ax=ax,
        units='trial_id',
        estimator=None
    )
    ax.set(xlabel="Time (s)", ylabel="Pupil size (%-change)", title=f'{subject}: {protocol.upper()}')
    ax.fill_between(
        (0, 3), min(ax.get_ylim()), max(ax.get_ylim()), alpha=0.2, color="k"
    )
    ax.grid()
    ax.legend(loc="lower right")
    fig.savefig(out_dir / f"processed_{protocol}_PLR_traces.png")
