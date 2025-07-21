# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:41:36 2024

@author: jmarti2
"""

# %%
import glob
import re
import os.path as op

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysilsub import problems
import seaborn as sns

plt.style.use("bmh")
plt.rcParams["font.size"] = 10


def tryint(s):
    try:
        return int(s)
    except:
        return s


def sort_files(s):
    return [tryint(c) for c in re.split("([0-9]+)", s)]


# %% Set these accordingly
power_path = r"C:\Users\jmarti2\OneDrive - University of Edinburgh\RetinaWISE\Calibration\Custom_Calibration_02092024\Power\*.csv"
spd_path = r"C:\Users\jmarti2\OneDrive - University of Edinburgh\RetinaWISE\Calibration\Custom_Calibration_02092024\SPD\*.csv"
out_path = r"C:\Users\jmarti2\OneDrive - University of Edinburgh\RetinaWISE\Calibration\Custom_Calibration_02092024"

# %% Process power measurements
files = glob.glob(power_path)
files = sorted(files, key=sort_files)

# Store the power data
vals = {
    "R1": [],
    "R2": [],
    "R3": [],
    "R4": [],
    "R5": [],
    "R6": [],
    "L1": [],
    "L2": [],
    "L3": [],
    "L4": [],
    "L5": [],
    "L6": [],
}

# Loop through the files
for f in files:
    # Extract primary and setting from the filename with regex
    p = re.findall("[RL][1-6](?=_)", f)[0]
    s = re.findall("[RL][1-6]_([1-9]+)", f)[0]
    df = pd.read_table(f)  # Load data
    val = df.loc[25].str.split(",").get(0)[1]  # Extract mean value
    val = pd.to_numeric(val)  # Convert to numeric
    # Negative power doesn not make sense so set to zero in this case
    if val < 0:
        val = 0
    vals[p].append(val)  # Save vals in appropriate dict

# Make labels for the ligt ratios, here as percent
ratios = np.hstack([[0, 1, 2], np.linspace(5, 100, 20)]).astype(int)

# Make DataFrame with ratios as index
power_vals = pd.DataFrame(vals, index=ratios)

# LED colors for plotting
colors = ["purple", "blue", "cyan", "green", "orange", "red"]

# Plot figure of power data
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
power_vals[["L1", "L2", "L3", "L4", "L5", "L6"]].plot(
    kind="line",
    ax=axs[0],
    color=colors,
    style="-o",
    lw=1,
    markeredgecolor="k",
    markeredgewidth=1,
)
power_vals[["R1", "R2", "R3", "R4", "R5", "R6"]].plot(
    kind="line",
    ax=axs[1],
    color=colors,
    style="-o",
    lw=1,
    markeredgecolor="k",
    markeredgewidth=1,
)

for ax in axs:
    ax.set_xlabel("Light Ratio")
    ax.set_ylabel("Optical Power (W)")

left_labels = [
    "L1 (420 $\lambda_{max}$)",
    "L2 (450 $\lambda_{max}$)",
    "L3 (470 $\lambda_{max}$)",
    "L4 (520 $\lambda_{max}$)",
    "L5 (590 $\lambda_{max}$)",
    "L6 (630 $\lambda_{max}$)",
]

right_labels = [
    "R1 (420 $\lambda_{max}$)",
    "R2 (450 $\lambda_{max}$)",
    "R3 (470 $\lambda_{max}$)",
    "R4 (520 $\lambda_{max}$)",
    "R5 (590 $\lambda_{max}$)",
    "R6 (630 $\lambda_{max}$)",
]
axs[0].legend(labels=left_labels)
axs[1].legend(labels=right_labels)
axs[0].set_title("Left Eye")
axs[1].set_title("Right Eye")
power_vals.index.name = "Light_Ratio"

# Save figure and csv
fig.savefig(op.join(out_path, "power_vals.png"))
power_vals.to_csv(op.join(out_path, "power_vals.csv"))

df = power_vals.reset_index().melt(id_vars='Light_Ratio', value_vars=['R1','R2','R3','R4','R5','R6','L1','L2','L3','L4','L5','L6'], var_name='LED')
df.loc[df.LED.str.startswith('L'), 'Eye'] = 'Left'
df.loc[df.LED.str.startswith('R'), 'Eye'] = 'Right'

# fig, axs = plt.subplots(1,2, figsize=(12,4))
# sns.lineplot(df.loc[df.Eye=='Left'], ax=axs[0], x='Light_Ratio', y='value', hue='LED', palette=colors, marker='o')
# sns.lineplot(df.loc[df.Eye=='Right'], ax=axs[1], x='Light_Ratio', y='value', hue='LED', palette=colors, marker='o')

# %% Process the spectra

# Get the file paths
files = glob.glob(spd_path)
files = sorted(files, key=sort_files)

# Separate store for left and right eye
left_spds = []
right_spds = []

# Loop over files
for f in files:
    # Extract Primary label using regex
    p = re.findall("[RL][1-6](?=_)", f)[0]
    # Load data
    df = pd.read_csv(
        f, sep=",", skiprows=38, header=None, encoding="unicode_escape"
    ).set_index(
        0
    )  # Wavelength set as index
    # Normalise spds and set column labels
    # Take sum of all jeti spectra and plot against optical power measurements  -  should be similar
    # Absolute units but unclear what they are
    # We validated our radiance measurements independantly with a different power meter and we get the same result
    norm_spds = df#.div(df.sum())  # sum instead of max!
    norm_spds.columns = ratios
    # Scale spds by power values and set index names
    scaled_spds = norm_spds.T#mul(power_vals[p]).T
    scaled_spds.columns.name = "Wavelength"
    scaled_spds.index.name = "Setting"
    # Add Primary as id column, set as index and reorder
    scaled_spds["Primary"] = int(p[1]) - 1
    scaled_spds = scaled_spds.set_index("Primary", append=True)
    scaled_spds = scaled_spds.reorder_levels(["Primary", "Setting"])
    # Add data to lists
    if "R" in p:
        right_spds.append(scaled_spds)
    elif "L" in p:
        left_spds.append(scaled_spds)

# Concat lists into dataframes
left = pd.concat(left_spds)
right = pd.concat(right_spds)

# Save csvs
left.to_csv(op.join(out_path, "left_spds.csv"))
right.to_csv(op.join(out_path, "right_spds.csv"))

# Make problems
ssp_left = problems.SilentSubstitutionProblem(
    left,
    [380, 780, 1],
    [100] * 6,
    ["violet", "blue", "cyan", "green", "orange", "red"],
    config=dict(calibration_units="W/[sr*sqm*nm]")
)
ssp_right = problems.SilentSubstitutionProblem(
    right,
    [380, 780, 1],
    [100] * 6,
    ["violet", "blue", "cyan", "green", "orange", "red"],
    config=dict(calibration_units="W/[sr*sqm*nm]")
)

# Plot spectra and gamut
left_fig = ssp_left.plot_calibration_spds_and_gamut()
right_fig = ssp_right.plot_calibration_spds_and_gamut()

# Save figures
left_fig.savefig(op.join(out_path, "left_spds.png"))
right_fig.savefig(op.join(out_path, "right_spds.png"))

#%%

left_df = left.sum(1).reset_index()
right_df = right.sum(1).reset_index()

fig, axs = plt.subplots(1,2, figsize=(12,4))
pal = ["violet", "blue", "cyan", "green", "orange", "red"]
sns.lineplot(left_df, x='Setting', hue='Primary', y=0, palette=pal, marker='o', ax=axs[0])
sns.lineplot(right_df, x='Setting', hue='Primary', y=0, palette=pal, marker='o', ax=axs[1])
for ax in axs:
    ax.set(ylim=(0,1))
