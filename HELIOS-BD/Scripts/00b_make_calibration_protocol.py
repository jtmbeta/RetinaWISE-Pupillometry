# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:46:36 2024

@author: jmarti2

Script makes a CSV protocol file for retina wise. The protocol will display 
each light at evenly spaced light ratios. Use in conjunction with a 
spectrometer or power meter.
"""

#%%
import copy
import functools
import random
import datetime

import numpy as np
import pandas as pd

today = str(datetime.datetime.now()).split()[0]

header = f"""
LR.exp;civibe_201;;;;;;;;;;;;;;;;;;;;;
Date;{today};;;;;;;;;;;;;;;;;;;;;
Author(s);JTM;;;;;;;;;;;;;;;;;;;;;
Photoreceptors;CIE tooolbox;;;;;;;;;;;;;;;;;;;;;
Calibration;Source;RetinaWISE_Edinburgh;;;;;;;;;;;;;;;;;;;;
Version;1;0;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;
Sampling time [ms];33;;;;;;;;;;;;;;;;;;;;;
Start delay [s];0;0;;;;;;;;;;;;;;;;;;;;
Temperature aquisition interval [tick];20;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;
Protocole:;;;;;;;;;;;;;;;;;;;;;;
NumSample;Label L;LED L1;LED L2;LED L3;LED L4;LED L5;LED L6;L L;M L;S L;g L;Label R;LED R1;LED R2;LED R3;LED R4;LED R5;LED R6;L R;M R;S R;g R
"""
eye = 'left'
delim = ';'
randomise = 0

def delimit(the_list):
    return functools.reduce(lambda x, y: str(x) + delim + str(y), the_list)

#%%
dark = [0,'dark',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
blank = [0] * 23

lines = []

primaries = range(6)
ratios = np.hstack([[0.01, 0.02], np.linspace(.05,1,20)])
# Shuffle order primary/ratio to avoid temperature issues


for eye in ['left', 'right']:
    for p in primaries:
        with open(f'./power_calibration_{eye}_{p}.csv', 'w') as f:
            f.writelines(header)
            f.write(delimit(dark))
            f.write('\n')
            for r in ratios:
                # Write the lines
                line = [0] * 23
                line[0] = 99
                if eye=='left':
                    line[1] = f'{p}-{int(round(r*100))}'
                    line[2+p] = f'{round(r, 2)}'
                elif eye=='right':
                    line[12] = f'{p}-{int(round(r*100))}'
                    line[13+p] = f'{round(r, 2)}'
                f.write(delimit(line))
                f.write('\n')
                # f.write(delimit(dark))
                # f.write('\n')
            long_dark = copy.copy(dark)
            long_dark[0] = 99
            f.write(delimit(long_dark))


