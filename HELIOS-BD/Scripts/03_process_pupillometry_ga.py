# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:01:43 2024

@author: jmarti2
"""
import pathlib as pl
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('bmh')
sns.set_context('talk')

# Check that subjects have identical time values. Currently 3011 and 3012 have issues.

datastore = pl.Path("Z:\Part B\Pupil_data")
files = list(datastore.rglob('*data_smooth_out\processed*PLRs_*.csv'))
exclude = ['2023', '1009', '1017', '1036',
           '1043', '1051', '1053', '3015', '3047', '1031']
ss_exclude = ['3039']
pipr_exclude = []

dfs = []
for fpath in files:
    subject = re.findall(r'PLRs_(\d{4})', str(fpath))[0]
    protocol = re.findall(r'processed_([A-Z]+)_', str(fpath))[0]
    print(f'**** {subject} : {protocol} ****')
    df = pd.read_csv(fpath)
    dfs.append(df)

#%%

gdf = pd.concat(dfs)
gdf['filt_pc_pupil'] = gdf['filt_pc_pupil'].sub(100)
gdf.subject = gdf.subject.astype('str')
gdf = gdf[~gdf.subject.isin(exclude)]
gdf = gdf.loc[gdf.pct_interpolated<.3]


# Add group infor
gdf.subject = gdf.subject.astype(str)
# Define the mapping of prefixes to groups
group_map = {
    '1': 'Control',
    '2': 'BD+Li',
    '3': 'BD-Li'
}
 
# Map the 'group' based on the first digit of the 'subject' column
gdf['group'] = gdf['subject'].apply(lambda x: group_map.get(x[0], 'Unknown'))

# aggregate
new = gdf.groupby(['group', 'subject', 'condition', 'time'],
                  as_index=False)[['filt_pc_pupil','filt_pupil']].mean()

pipr = new.loc[new.condition.isin(['red', 'blue'])].reset_index(drop=True)
ss = new.loc[new.condition.isin(['lms', 'mel'])].reset_index(drop=True)

titles = gdf[['group', 'subject']].drop_duplicates().groupby('group').count()

#%%
fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout='tight', sharex=True)
for i, group in enumerate(group_map.values()):
    
    # Plot the difference waves
    # agg = pipr.groupby(['group','condition','time'])['filt_pc_pupil'].mean()
    # difference_wave = agg.loc[group, 'red'] - agg.loc[group, 'blue'] + 100
    # difference_wave.plot(ax=axs[0, i], color='k', lw=2, label='difference wave')

    # agg = ss.groupby(['group','condition','time'])['filt_pc_pupil'].mean()
    # difference_wave = agg.loc[group, 'mel'] - agg.loc[group, 'lms'] + 100
    # difference_wave.plot(ax=axs[1, i], color='k', lw=2, label='difference wave')
    ss=ss.loc[~ss.subject.isin(ss_exclude)]

    sns.lineplot(
        data=pipr.loc[pipr.group == group],
        x="time",
        y="filt_pc_pupil",
        hue='condition',
        errorbar=None,
        palette={'red': 'tab:red', 'blue': 'tab:blue'},
        ax=axs[0, i],
        units='subject',
        estimator=None,
        lw=.3,
        legend=False
    )

    sns.lineplot(
        data=ss.loc[ss.group == group],
        x="time",
        y="filt_pc_pupil",
        hue='condition',
        errorbar=None,
        palette={'lms': 'tab:green', 'mel': 'tab:blue'},
        ax=axs[1, i],
        units='subject',
        estimator=None,
        lw=.3,
        legend=False
    )
    
    sns.lineplot(
        data=pipr.loc[pipr.group == group],
        x="time",
        y="filt_pc_pupil",
        hue='condition',
        errorbar=None,
        palette={'red': 'tab:red', 'blue': 'tab:blue'},
        ax=axs[0, i],
        #units='subject',
        #estimator=None,
        lw=2
    )
    sns.lineplot(
        data=ss.loc[ss.group == group],
        x="time",
        y="filt_pc_pupil",
        hue='condition',
        errorbar=None,
        palette={'lms': 'tab:green', 'mel': 'tab:blue'},
        ax=axs[1, i],
        #units='subject',
        #estimator=None,
        lw=2
    )
    axs[0, 0].set_title('Control')
    axs[0, 1].set_title('BD+Li')
    axs[0, 2].set_title('BD-Li')
    

for ax in axs[0]:
    ax.set_ylim(-70, 20)

for ax in axs[1]:
    ax.set_ylim(-30, 20)

for ax in axs.flatten():
    ax.set(xlabel="Time (s)", ylabel="Pupil size (%-change)")
    ax.fill_between(
        (0, 3), min(ax.get_ylim()), max(ax.get_ylim()), alpha=0.2, color="k"
    )
    #ax.grid()
    ax.legend(title=None)

axs[0, 2].legend_.set_bbox_to_anchor((1,1))
axs[1, 2].legend_.set_bbox_to_anchor((1,1))


axs[0, 0].get_legend().remove()
axs[0, 1].get_legend().remove()
axs[1, 0].get_legend().remove()
axs[1, 1].get_legend().remove()
plt.show()
fig.savefig('all_pupil_responses.svg', bbox_inches='tight')

#%% PIPR

pivot = pipr.pivot(index=['group', 'subject', 'time'],
                   columns='condition', values='filt_pc_pupil')
pivot['difference'] = pivot.red - pivot.blue
pivot = pivot.reset_index()
# sns.lineplot(
#     data=pivot,
#     x="time",
#     y="difference",
#     # hue='condition',
#     errorbar='se',
#     # ax=axs[1, i],
#     hue='group',
#     # units='subject',
#     # estimator=None,
#     lw=.3
# )

diff = pivot[pivot.time > 3].groupby(['group', 'subject'])[
    'difference'].mean().reset_index()
fig,ax=plt.subplots()
sns.barplot(diff, x='group',y='difference', color='gray',ax=ax)
ax.set(ylabel='Net PIPR (%)',xlabel='')
fig.savefig('net_PIPR.svg',bbox_inches='tight')

diff.to_csv('pipr_ave_diff.csv')

t2maxcon = pipr.set_index(['group','subject','time']).groupby(['group', 'subject'])['filt_pc_pupil'].idxmin().apply(
    lambda x: x[-1]).reset_index(name='min_pupil_time')
t2maxcon.to_csv('pipr_t2maxcon.csv')

conamp = pipr.groupby(['group', 'subject'])['filt_pc_pupil'].min().reset_index(name='constriction_amplitude')
conamp.to_csv('pipr_conamp.csv')

# Age
piprage = gdf.loc[gdf.condition.isin(['red','blue'])]
piprage = piprage[['group','subject','age','baseline','trial']].drop_duplicates().reset_index()
piprage = piprage.groupby(['group', 'subject','age'], as_index=False).mean()

piprage.merge(t2maxcon).merge(diff)
#%% Silent substitution

pivot = ss.pivot(index=['group', 'subject', 'time'],
                   columns='condition', values='filt_pc_pupil')
pivot['difference'] = pivot.lms - pivot.mel
pivot = pivot.reset_index()

diff = pivot[pivot.time > 3].groupby(['group', 'subject'])[
    'difference'].mean().reset_index()
fig,ax=plt.subplots(figsize=(3,3))
sns.barplot(diff, x='group',y='difference', color='gray',ax=ax)
ax.set(ylabel='Net PIPR (%)',xlabel='')
#diff.to_csv('pipr_ave_diff.csv')


