# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:56:47 2024

@author: jmarti2

This script creates the LMS protocol for the HELLIOS-BD project. 

The protocol begins with a 4 minute adaptation to the background spectrum, followed
by 16 pulses of LMS-directed contrast. 

Stimuli are 3-s pulses of light windowed by a half cosine ramp (500 ms each 
side). There is a 1 s baseline and a 16 s measurement period surrounding the
stimulus. Additionally, there is a random jitter of 0 - 3 s between each 
stimulus.

The silent substitution stimulus is designed to achieve high contrast on cone 
photoreceptors whilst silencing melanopsin. A 'pre-made' solution is first
identified and then optimised to account for the age of the individual.

For further information and rationale, see:

McAdams, H., Igdalova, A., Spitschan, M., Brainard, D. H., &; Aguirre, G. K. 
    (2018). Pulses of melanopsin-directed contrast produce highly reproducible
    pupil responses that are insensitive to a change in background radiance. 
    Investigative Ophthalmology and Visual Science, 59(13), 5615–5626.
    https://doi.org/10.1167/iovs.18-25219

"""
import os
import os.path as op
import pickle
import functools
import datetime
import random

from scipy import signal
import pandas as pd
from pysilsub import problems
from pysilsub import observers
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

# %%
for PARTICIPANT_AGE in range(20,65):

# %%
    
    # Define the observer
    obs = observers.ColorimetricObserver(age=PARTICIPANT_AGE, field_size=10)
    
    # Set up the problem
    sspr = problems.SilentSubstitutionProblem(
        calibration=r"C:\Users\jmarti2\OneDrive - University of Edinburgh\RetinaWISE\Calibration\Custom_Calibration_02092024\right_spds.csv",
        calibration_wavelengths=[380, 780, 1],
        primary_resolutions=[100] * 6,
        primary_colors=["violet", "blue", "cyan", "green", "orange", "red"],
        observer=obs,
        name="RetinaWISE",
        config=dict(calibration_units="Energy"),
    )
    sspr.bounds = [(0.05, 0.95)] * 6
    
    # This is a pre-baked solution that we are going to optimize to the individual
    solution = np.array(
        [
            0.05,
            0.07316009,
            0.46901663,
            0.05,
            0.05,
            0.83928922,
            0.94836887,
            0.06315093,
            0.05,
            0.38591947,
            0.95,
            0.89897302,
        ]
    )
    
    
    # Define the problem
    sspr.target = ["sc", "mc", "lc"]
    sspr.silence = ["mel"]
    sspr.ignore = ["rh"]
    sspr.background = None
    sspr.target_contrast = 1.5
    x0 = np.array([random.random() for i in range(12)])
    sr = sspr.optim_solve(
        x0=solution, global_search=False, **{"options": {"niter": 200}}
    )
    fig = sspr.plot_solution(sr.x)
    sspr.print_photoreceptor_contrasts(sr.x, "simple")
    nom_con = sspr.get_photoreceptor_contrasts(sr.x, "simple")
    print(f"Background: {sr.x[0:6]}")
    print(f"Modulation: {sr.x[6:]}")
    
    # Make an output directory
    out_dir = rf"C:\Users\jmarti2\OneDrive - University of Edinburgh\RetinaWISE\Protocols\HELIOS-BD\LMS\{PARTICIPANT_AGE}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save the optimization result
    with open(op.join(out_dir, r"optim_result.pkl"), "wb") as output_file:
        pickle.dump(sr, output_file)
    
    # Save the solution plot
    fig.savefig(op.join(out_dir, "solution_figure.svg"))
    
    # Save nominal contrast
    nom_con.to_csv(op.join(out_dir, "nominal_contrasts.csv"))
    
    # Save spectra
    bg = sspr.predict_multiprimary_spd(sr.x[0:6])
    mod = sspr.predict_multiprimary_spd(sr.x[6:])
    bg.name = "background"
    mod.name = "modulation"
    df = pd.concat([bg, mod], axis=1)
    df.to_csv(op.join(out_dir, "nominal_spectra.csv"))
    
    # %% Create protocol
    
    
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
    jitter = (
        [1, "jitter", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "jitter"]
        + list(sr.x[0:6])
        + [0, 0, 0, 0]
    )
    # The baseline period (the background spectrum)
    baseline = (
        [50, "baseline", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "baseline"]
        + list(sr.x[0:6])
        + [0, 0, 0, 0]
    )
    # The background spectrum
    bg = [
        [1, "baseline", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "baseline"]
        + list(sr.x[0:6])
        + [0, 0, 0, 0]
        for i in range(150)
    ]
    # The lms spectrum
    lms = [
        [1, "lms", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "lms"]
        + list(sr.x[6:])
        + [100, 100, 100, 0]
        for i in range(150)
    ]
    # The measurement period
    measurement = (
        [800, "measurement", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "measurement"]
        + list(sr.x[0:6])
        + [0, 0, 0, 0]
    )
    # The adaptation period
    adapt = (
        [12000, "adapt", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "adapt"]
        + list(sr.x[0:6])
        + [0, 0, 0, 0]
    )
    
    # Make DataFrames
    df_bg = pd.DataFrame(bg, columns=cols)
    df_lms = pd.DataFrame(lms, columns=cols)
    
    # %% Apply cosine ramp
    
    mod = (df_bg.loc[:, "LED R1":"LED R6"] - df_lms.loc[:, "LED R1":"LED R6"]).mul(
        wf, axis=0
    )
    df_lms.loc[:, "LED R1":"LED R6"] = df_bg.loc[:, "LED R1":"LED R6"].sub(mod)
    
    # Save values in a list
    lms_list = df_lms.values.tolist()
    # %%
    
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
    
    # Create the protocol file
    with open(op.join(out_dir, f"HELIOS_LMS_SS_{PARTICIPANT_AGE}.csv"), "w") as f:
        f.writelines(header)
        f.write(delimit(sequence))
        f.write("\n")
        f.write(delimit(adapt))
        f.write("\n")
        for i in range(15):  # Number of repeats
            # Write sequence breaker
            f.write(delimit(sequence))
            f.write("\n")
    
            # Write jitter
            jitter[0] = random.randrange(1, 150)
            f.write(delimit(jitter))
            f.write("\n")
    
            # Write baseline period
            f.write(delimit(baseline))
            f.write("\n")
    
            # Write the LMS stimulus
            for i in lms_list:
                f.write(delimit(i))
                f.write("\n")
    
            # Write measurment period
            f.write(delimit(measurement))
            f.write("\n")
    
    
    # %% Create measurement file
    
    # This protocol file can be used to obtain measurements of the background and
    # modulation spectra before the experiment (using the milk glass approach)
    
    # Create the protocol file
    with open(op.join(out_dir, "./HELIOS_LMS_2_measurement_file.csv"), "w") as f:
        f.writelines(header)
    
        # Write sequence breaker
        f.write(delimit(sequence))
        f.write("\n")
    
        # Background
        f.write(delimit(bg[0]))
        f.write("\n")
    
        # Write sequence breaker
        f.write(delimit(sequence))
        f.write("\n")
    
        # Modulation
        f.write(delimit(lms[0]))
        f.write("\n")

        # Write sequence breaker
        f.write(delimit(sequence))
        f.write("\n")
