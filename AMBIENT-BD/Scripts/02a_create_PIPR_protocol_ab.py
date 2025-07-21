# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:44:03 2024

@author: jmarti2

This script creates the PIPR protocol for the HELLIOS-BD project. 

The protocol comprises 6 red and 6 blue light stimli presented alternately. It
should be administered after 10 minutes of dark adaptation. 

Stimuli are 3-s pulses of light windowed by a half cosine ramp (500 ms each 
side). There is a 1 s baseline and a 16 s measurement period surrounding the
stimulus. Additionally, there is a random jitter of 0 - 3 s between each 
stimulus.

The intensity of the red and blue stimuil are matched for quantal radiance
using an optimization procedure that accounts for lens transmittance given the
age of the observer (blue stimuli are affected more in this regard).

For further information and rationale, see:

McAdams, H., Igdalova, A., Spitschan, M., Brainard, D. H., &; Aguirre, G. K. 
    (2018). Pulses of melanopsin-directed contrast produce highly reproducible
    pupil responses that are insensitive to a change in background radiance. 
    Investigative Ophthalmology and Visual Science, 59(13), 5615–5626.
    https://doi.org/10.1167/iovs.18-25219
    
Martinsons C, Behar-Cohen F, Bergen T, et al. Reconsidering the spectral
    distribution of light: Do people perceive watts or photons? Lighting 
    Research & Technology. 2024;0(0). doi:10.1177/14771535241246060
    
"""

import os.path as op
import os
import datetime
import functools
import random

import matplotlib.pyplot as plt
from pysilsub import problems, observers
from scipy import signal
from scipy import optimize
from scipy import constants
import numpy as np
import pandas as pd


# %%
# Set the participant age accordingly.
for PARTICIPANT_AGE in range(20,65):

    # This is the target log quantal radiance, chosen because it is achieveable for
    # ages ranged 20 - 65 after lens filtering
    target_log_quanta = 17.
    
    # %%
    right_cal = r"C:\Users\jmarti2\OneDrive - University of Edinburgh\Oculox\WP2_JTM\Calibration\20250519_Custom_Calibration_JTM\right_spds.csv"
    right_ssp = problems.SilentSubstitutionProblem(
        calibration=right_cal,
        calibration_wavelengths=[380, 781, 1],
        primary_resolutions=[100] * 6,
        primary_colors=["violet", "blue", "cyan", "green", "orange", "red"],
        name="RetinaWISE (right)",
    )
    
    # %%  Get lens transmittance appropriate to age
    
    ld = (
        observers.get_lens_density_spectrum(PARTICIPANT_AGE)
        .reindex(range(380, 781, 1))[::-1]
        .interpolate()[::-1]
    )
    t = 10 ** (-ld)  # transmittance
    
    # %% optimize
    
    
    def energy_to_quanta(spectrum):
        """Convert radiance spectrum to quantal radiance."""
        return spectrum.div(constants.h * constants.c).mul(1e-9 * spectrum.index)
    
    
    # 17.65 log quanta is achievable for age ranges 20 - 65
    def objective_quanta(x0, ssp, t, led):
        s = ssp.predict_primary_spd(led, x0[0]).mul(t)
        sq = energy_to_quanta(s)
        return abs(target_log_quanta - np.log10(sq.sum())) ** 2
    
    
    def plot_solution(rs, bs, ssp, t):
        # Filter the spectra
        b = ssp.predict_primary_spd(2, bs)
        r = ssp.predict_primary_spd(5, rs)
        bfilt = energy_to_quanta(b.mul(t))
        rfilt = energy_to_quanta(r.mul(t))
    
        plt.bar([1, 2], [bfilt.sum(), rfilt.sum()], color=["cyan", "red"])
        plt.show()
        b.plot(ls=":", color="cyan")
        r.plot(ls=":", color="red")
        # bfilt.plot(ls="--", color="cyan")
        # rfilt.plot(ls="--", color="red")
        plt.show()
    
    
    # Perofrm the optimization for blue
    res_blue = optimize.minimize(
        fun=objective_quanta,
        x0=[0.5],  # Initial high guess for the settings works well
        args=(right_ssp, t, 2),
        bounds=[(0.05, 1.0)],
    )
    
    # Perofrm the optimization for red
    res_red = optimize.minimize(
        fun=objective_quanta,
        x0=[0.5],  # Initial high guess for the settings works well
        args=(right_ssp, t, 5),
        bounds=[(0.05, 1.0)],
    )
    
    # Assign the LED input values
    bs = res_blue.x[0]
    rs = res_red.x[0]
    
    # Plot the solution
    plot_solution(rs, bs, right_ssp, t)
    
    # Make output directory
    #out_dir = rf"C:\Users\jmarti2\OneDrive - University of Edinburgh\RetinaWISE\Protocols\HELIOS-BD\PIPR\{PARTICIPANT_AGE}"
    out_dir = rf"C:\Users\jmarti2\OneDrive - University of Edinburgh\Oculox\WP2_JTM\Protocols\PIPR\{PARTICIPANT_AGE}"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Save nominal spectra
    red = right_ssp.predict_primary_spd(5, rs)
    blue = right_ssp.predict_primary_spd(2, bs)
    red.name = "red"
    blue.name = "blue"
    df = pd.concat([red, blue], axis=1)
    df.to_csv(op.join(out_dir, "nominal_spectra.csv"))
    
    # Print LED input values
    print("The LED settings: ")
    print(f"\tBlue: {bs}")
    print(f"\tRed: {rs}")
    
    print("The LED radiant powers: ")
    print(f"\tBlue: {blue.sum()}")
    print(f"\tRed: {red.sum()}")
    
    print("The LED quanta: ")
    print(f"\tBlue: {energy_to_quanta(blue.mul(t)).sum()}")
    print(f"\tRed: {energy_to_quanta(red.mul(t)).sum()}")
    
    print("The LED log quanta: ")
    print(f"\tBlue: {np.log10(energy_to_quanta(blue.mul(t)).sum())}")
    print(f"\tRed: {np.log10(energy_to_quanta(red.mul(t)).sum())}")
    
    # %% Make protocol file
    
    # The sampling time of the RetinaWISE device. Can be as low as 5 ms
    sampling_time = 20
    
    # A 1-s cosine window for the stimulus
    cw = signal.windows.cosine(50)
    
    # The complete waveform with 2 s full power in the middle
    wf = np.hstack([cw[:25], np.ones(100), cw[25:]])
    
    # Columns of the protocol file
    cols = [
        "NumSample",
        "Label L",
        "LED L1",
        "LED L2",
        "LED L3",
        "LED L4",
        "LED L5",
        "LED L6",
        "L L",
        "M L",
        "S L",
        "g L",
        "Label R",
        "LED R1",
        "LED R2",
        "LED R3",
        "LED R4",
        "LED R5",
        "LED R6",
        "L R",
        "M R",
        "S R",
        "g R",
    ]
    
    # This is the sequence breaker
    sequence = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    # Random jitter. Later we will choose a random number for NumSample
    jitter = [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        "jitter",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    # The baseline period
    baseline = [
        50,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        "baseline",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    # The red stimulus
    reds = [
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "red",
            0,
            0,
            0,
            0,
            0,
            rs,
            100,
            0,
            0,
            0,
        ]
        for i in range(150)  # 3 s
    ]
    # The blue stimulus
    blues = [
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "blue",
            0,
            0,
            bs,
            0,
            0,
            0,
            0,
            0,
            100,
            0,
        ]
        for i in range(150)  # 3 s
    ]
    # 16 s measurement period
    measurement = [
        800,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        "measurement",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    
    # Make DataFrames
    df_blue = pd.DataFrame(blues, columns=cols)
    df_red = pd.DataFrame(reds, columns=cols)
    
    # Apply the cosine ramp
    df_blue["LED R3"] = df_blue["LED R3"].mul(wf)  # Blue stimulus
    df_red["LED R6"] = df_red["LED R6"].mul(wf)  # Red stimulus
    
    # Turn into list of list
    blue_list = df_blue.values.tolist()
    red_list = df_red.values.tolist()
    
    # Delimiter used for protocol files
    delim = ";"
    
    
    # Function to delimit lists for protocol files
    def delimit(the_list):
        return functools.reduce(lambda x, y: str(x) + delim + str(y), the_list)
    
    
    # Date the protocol files was created
    today = str(datetime.datetime.now()).split()[0]
    
    # The top-matter for the protocol file, includes sampling_time
    # variable
    header = f"""LR.exp;civibe_201;;;;;;;;;;;;;;;;;;;;;
    Date;{today};;;;;;;;;;;;;;;;;;;;;
    Author(s);JTM;;;;;;;;;;;;;;;;;;;;;
    Photoreceptors;CIE tooolbox;;;;;;;;;;;;;;;;;;;;;
    Calibration;Source;RetinaWISE_Edinburgh;;;;;;;;;;;;;;;;;;;;
    Version;1;0;;;;;;;;;;;;;;;;;;;;
    ;;;;;;;;;;;;;;;;;;;;;;
    Sampling time [ms];{sampling_time};;;;;;;;;;;;;;;;;;;;;
    Start delay [s];0;0;;;;;;;;;;;;;;;;;;;;
    Temperature aquisition interval [tick];20;;;;;;;;;;;;;;;;;;;;;
    ;;;;;;;;;;;;;;;;;;;;;;
    Protocole:;;;;;;;;;;;;;;;;;;;;;;
    NumSample;Label L;LED L1;LED L2;LED L3;LED L4;LED L5;LED L6;L L;M L;S L;g L;Label R;LED R1;LED R2;LED R3;LED R4;LED R5;LED R6;L R;M R;S R;g R
    """
    trial_id = 0
    # Create the protocol file
    with open(op.join(out_dir, f"./AMBIENT_PIPR_{PARTICIPANT_AGE}.csv"), "w") as f:
        f.writelines(header)
        for trialnum in range(1,7):  # Six of each stimuli
            # Write sequence breaker
            #f.write(delimit(sequence))
            #f.write("\n")
            trial_id += 1
            # Write jitter
            jitter[0] = random.randrange(1, 150)
            jitter[12] = f"{trial_id}-red-jitter"
            f.write(delimit(jitter))
            f.write("\n")
    
            # Write baseline period
            baseline[12] = f"{trial_id}-red-baseline"
            f.write(delimit(baseline))
            f.write("\n")
    
            # Write the RED stimulus
            for i in red_list:
                i[12] = f"{trial_id}-red-stimulation"
                f.write(delimit(i))
                f.write("\n")
    
            # Write measurment period
            measurement[12] = f"{trial_id}-red-measurement"
            f.write(delimit(measurement))
            f.write("\n")
    
            # Write sequence breaker
            #f.write(delimit(sequence))
            #f.write("\n")
            trial_id += 1

            # Write jitter
            jitter[0] = random.randrange(1, 150)
            jitter[12] = f"{trial_id}-blue-jitter"
            f.write(delimit(jitter))
            f.write("\n")
    
            # Write baseline period
            baseline[12] = f"{trial_id}-blue-baseline"
            f.write(delimit(baseline))
            f.write("\n")
    
            # Write the BLUE stimulus
            for i in blue_list:
                i[12] = f"{trial_id}-blue-stimulation"
                f.write(delimit(i))
                f.write("\n")
    
            # Write measurment period
            measurement[12] = f"{trial_id}-blue-measurement"
            f.write(delimit(measurement))
            f.write("\n")
    

