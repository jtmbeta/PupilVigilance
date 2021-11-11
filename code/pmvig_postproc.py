#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data wrangling and plotting script for vigilance experiment
===========================================================

@author: jtm

"""

# %% Imports  and config

from scipy.stats import norm
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
import pandas as pd
import seaborn as sns
import numpy as np

from vig_func import mad

sns.set(context='paper', style='ticks', font='helvetica', font_scale=2.1)

# %% Get all pupil data into a single DataFrame

# Load store and loop to get all subjects
s = pd.HDFStore('../pmvig/analysis/event_ranges_50hz.h5')
df_pupil = pd.DataFrame()
for sub in s.keys():
    p = s[sub]
    # Exclude subjects with too much interpolated data
    pct_interp = p.groupby('event')['masked'].mean().mean()
    if pct_interp > .35:
        p['BadSubj'] = 1
    else:
        p['BadSubj'] = 0
    df_pupil = df_pupil.append(p)
s.close()

# Colum with time relative to button press
df_pupil['Time'] = df_pupil.index.get_level_values(1) * (1/50) - 1
df_pupil = df_pupil.reset_index()
df_pupil.to_csv('../pmvig/analysis/all_pupil_event_ranges.csv', index=False)

# %% Process and plot pupil traces for Block

# Drop the bad subjects / trials
pupil_ex = (df_pupil.loc[((df_pupil['pct_interp>25'] == 0)
                          & (df_pupil['base_blink'] == 0)
                          & (df_pupil['BadSubj'] == 0))])

# Idxs for kept trials
keeps = pupil_ex[['Subject', 'event']].drop_duplicates().reset_index(drop=True)

# Kepp track of percent exclusions
n_trials = len(df_pupil[df_pupil['BadSubj'] == 0].groupby(
    ['Subject', 'event'], as_index=False).count())
pct_exclude = (1 - (len(keeps) / n_trials))

# Percent interpolated for used trials
pct_interp = pupil_ex.groupby('Subject')['masked'].mean().mean()
print(f'Percentage of interpolated data (used trials): {pct_interp}\n')

# Percentage of excluded trials
print(f'Percentage of excluded trials: {pct_exclude}\n')

# Get the baselines
s = pd.HDFStore('../pmvig/analysis/event_baselines.h5')
baselinez = pd.DataFrame()
for sub in s.keys():
    p = s[sub]
    p['Subject'] = sub[1:]
    baselinez = baselinez.append(p)
s.close()

# Get rid of the trials we want to keep
baselinez_ex = baselinez.reset_index().merge(keeps)
baselinez_ex[['Block', 'TrialGroup']] = (baselinez_ex[['Block', 'TrialGroup']]
                                         .astype('int'))

# Get the evoked responses
evoked = (pupil_ex.loc[pupil_ex.onset > 25]  # Time of key press
          .groupby(by=['Subject', 'Block', 'TrialGroup'],
                   as_index=False)['p_pc']  # Percent modulation
          .mean())

# Get the evoked responses
bls = (baselinez_ex.groupby(by=['Subject', 'Block', 'TrialGroup'],
                            as_index=False)['pz']  # Z score
       .mean())

# Get averages per subject / block
data = pupil_ex.groupby(['Subject', 'Block', 'onset'], as_index=False).mean()
masked = data.groupby('onset')['masked'].mean()  # So we can plot it

# Perform cluster-based permutation stats on the pupil traces
print('*** Cluster permutation tests for response-locked pupil traces ***\n')

# Aggregate pupil data and pivot
pvt = data.pivot_table(values='p_pc',
                       index=['Block', 'Subject'],
                       columns='onset')

print('\n> Block: 1')
res1 = permutation_cluster_1samp_test(pvt.loc[1].values)
print(f'> Significance and slice indices: {res1[1:3]}')
print('\n> Block: 2')
res2 = permutation_cluster_1samp_test(pvt.loc[2].values)
print(f'> Significance and slice indices: {res2[1:3]}')
print('\n> Block: 3')
res3 = permutation_cluster_1samp_test(pvt.loc[3].values)
print(f'> Significance and slice indices: {res3[1:3]}')
print('\n> Block: F test')
res4 = permutation_cluster_test(
    [pvt.loc[1].values, pvt.loc[2].values, pvt.loc[3].values])
print(f'> Significance and slice indices: {res4[1:3]}')

# Plot pupil data
palette = sns.color_palette()[0:3]
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# Plot pupil data
sns.lineplot(data=data, x='onset', y='p_pc',
             hue='Block', ax=ax, palette=palette, ci=68)
ax.set(xlabel='Time (s)',
       ylim=(-3, 12),
       ylabel='Pupil modulation (%)')
ax.axhline(0, 0, 1, c='k', ls='--')
ax.axvline(50, 0, 1, c='k', ls='--')
ax.text(90, .5, '$P$ < .05')
ax.set_title('Button-locked')

# Indicate the average time of the target stimulus (mean RT = 475 / (1000/50))
ax.text(-.85, 3, 'Average time\n of stimulus', fontsize=14)
ax.vlines(26.25, 0, 2.8, linestyle='solid', color='k')

# Arrange ticks / labels on x-axis
ax.set_xticks(np.arange(0, 175, 25))
ax.set_xticklabels([str(val) for val in np.arange(-1., 2.1, .5)])

# Coloured bars to show significant clusters. The extent of these bars is
# determined by the slice indices of the significant permutation tests
plt.hlines(-.5, 78, 150, lw=5, alpha=.5, color=palette[0])
plt.hlines(-1, 82, 120, lw=5, alpha=.5, color=palette[1])
plt.hlines(-1.5, 80, 150, lw=5, alpha=.5, color=palette[2])

# Plot percent interpolated data, which explains the dip in the traces
newax = ax.twinx()
newax.plot(masked, c='k', ls=':')
newax.set(ylabel='Interpolated data (%)',
          ylim=(0, 1))

sns.despine(offset=10, right=False, left=True,
            top=True, bottom=True, trim=False, ax=newax)
sns.despine(offset=10, trim=True, ax=ax)

fig.savefig('../pmvig/analysis/manuscript_figs/pmvig_pupil_block.tiff',
            dpi=100, bbox_inches='tight')

# %% Plot pupil scalars and performance metrics

# Get the RT data
s = pd.HDFStore('../pmvig/analysis/RTs.h5')
df_RT = pd.DataFrame()
for sub in s.keys():
    rts = s[sub]
    rts = rts.reset_index().rename(columns={'event': 'event_label'})
    rts['event'] = rts.index
    df_RT = df_RT.append(rts)
s.close()
df_RT = df_RT.reset_index()

# Drop the bad subjects and edge cases
df_RT = df_RT.loc[~df_RT.Subject.isin(['sub007', 'sub009'])]
df_RT = df_RT.loc[df_RT.RT < 50000]  # 3 RTs longer than this

# Aggregate
aggregations = {
    '1/RT': 'mean',
    'RT': 'mean',
    'Lapse': 'sum'
}

ag = (df_RT.groupby(['Subject', 'Block', 'TrialGroup'])
      .aggregate(aggregations)
      .reset_index())

# Plot performance and pupil measures
fig, axs = plt.subplots(2, 2, figsize=(8, 7))
axs = axs.flatten()
sns.pointplot(
    data=ag, x='TrialGroup', y='1/RT',
    units='Subject', hue='Block',
    dodge=.3, ax=axs[0], markers='s'
)
axs[0].legend([], [], frameon=False)

sns.pointplot(
    data=ag, x='TrialGroup', y='Lapse',
    units='Subject', hue='Block',
    dodge=.3, ax=axs[1], markers='s'
)
axs[1].legend([], [], frameon=False)
axs[1].set(ylim=(0, 4),
           ylabel='Lapses')

sns.pointplot(
    data=bls, x='TrialGroup', y='pz',
    units='Subject', hue='Block',
    dodge=.3, ax=axs[2], markers='s'
)
axs[2].set(ylabel='Baseline pupil size (z)')
axs[2].legend([], [], frameon=False)

sns.pointplot(
    data=evoked, x='TrialGroup', y='p_pc',
    units='Subject', hue='Block',
    dodge=.3, ax=axs[3], markers='s'
)
axs[3].set(ylabel='Pupil modulation (%)',
           ylim=(0, 20))

plt.tight_layout()
fig.savefig(
    '../pmvig/analysis/manuscript_figs/performance_and_pupil_figure.tiff',
    dpi=300)

# Save behavioral data for repeated measures analysis in SPSS / JASP
ag[['Block', 'TrialGroup']] = ag[['Block', 'TrialGroup']].astype('str')
performance_stats = ag.pivot(index='Subject',
                             columns=['TrialGroup', 'Block'],
                             values=['1/RT', 'RT', 'Lapse'])
performance_stats.columns = performance_stats.columns.map(
    '|'.join).str.strip('|')
performance_stats.to_csv('../pmvig/analysis/rm_behavioral.csv')

# Save pupil data for repeated measures analysis in SPSS / JASP
evoked[['Block', 'TrialGroup']] = evoked[['Block', 'TrialGroup']].astype('str')
bls[['Block', 'TrialGroup']] = bls[['Block', 'TrialGroup']].astype('str')
pupil_stats = evoked.pivot(index='Subject',
                           columns=['Block', 'TrialGroup'], values=['p_pc'])
pupil_stats.columns = pupil_stats.columns.map('|'.join).str.strip('|')
bls_pivot = bls.pivot(index='Subject',
                      columns=['Block', 'TrialGroup'], values=['pz'])
bls_pivot.columns = bls_pivot.columns.map('|'.join).str.strip('|')
pupil_stats = pupil_stats.merge(bls_pivot, on='Subject')
pupil_stats.to_csv('../pmvig/analysis/rm_pupil.csv')

# %% Fastest / slowest 20 % RTs

# Get the pupil data for the fastest and slowest 20 % trials
fastest_slowest_pupil = pupil_ex.loc[pupil_ex.Quintile.isin([1, 5])]
baselinez_ex['Quintile'] = baselinez_ex['Quintile'].astype('int')
fastest_slowest_bls = baselinez_ex.loc[baselinez_ex.Quintile.isin([1, 5])]

# Describe the fastest and slowest RTs
df_RT.loc[df_RT.Quintile.isin([1, 5])].groupby('Quintile')['RT'].describe()

# Plot baselines and pupil traces
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot baseline pupil size for fastest and slowest
sns.barplot(data=fastest_slowest_bls, x='Quintile',
            units='Subject', y='pz', ax=axs[0])
axs[0].axhline(0, 0, 1, c='k', ls='--')
axs[0].set(xlabel='',
           ylabel='Baseline pupil size (z)')
axs[0].set_xticklabels(['Fastest 20% RT', 'Slowest 20% RT'])
axs[0].text(.5, .1, 'n.s.\n($p$ > .05)', ha='center', va='center')

for tick in axs[0].xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)

# Perform cluster-based permutation stats on the pupil traces
print('*** Cluster permutation tests ***\n')

new = (fastest_slowest_pupil.merge(
    fastest_slowest_bls[['Quintile', 'Subject', 'event']],
    on=['Subject', 'event', 'Quintile'])
    .groupby(['Subject', 'onset', 'Quintile'], as_index=False)
    .mean())

new.Quintile = new.Quintile.replace(1, 'fastest 20%')
new.Quintile = new.Quintile.replace(5, 'slowest 20%')
new = new.rename(columns={'Quintile': 'RT'})


agp = new.groupby(['Subject', 'RT', 'onset'], as_index=False).mean()
pvt = agp.pivot_table(values='p_pc', index=['RT', 'Subject'], columns='onset')

print('\n> Fastest 20 % RTs')
res1 = permutation_cluster_1samp_test(pvt.loc['fastest 20%'].values)
print(f'> Significance and slice indices: {res1[1:3]}')
print('\n> Slowest 20 % RTs')
res2 = permutation_cluster_1samp_test(pvt.loc['slowest 20%'].values)
print(f'> Significance and slice indices: {res2[1:3]}')
print('\n> Slowest 20 % - Fastest 20 % (difference)')
res_diff = permutation_cluster_1samp_test(
    pvt.loc['slowest 20%'].values - pvt.loc['fastest 20%'].values)
print(f'> Significance and slice indices: {res_diff[1:3]}')

# Plot the pupil traces for fastest and slowest
sns.lineplot(data=new, x='onset', y='p_pc',
             hue='RT', ax=axs[1], legend=False, ci=68)
axs[1].set(xlabel='Time (s)',
           ylim=((-4, 12)),
           ylabel='Pupil modulation (%)')
axs[1].axhline(0, 0, 1, c='k', ls='--')
axs[1].axvline(50, 0, 1, c='k', ls='--')
axs[1].text(80, -2.5, '$P$ < .05')
axs[1].set_title('Button-locked')

# Arrange ticks / labels on x-axis
axs[1].set_xticks(np.arange(0, 175, 25))
axs[1].set_xticklabels([str(val) for val in np.arange(-1., 2.1, .5)])

# Show the average time of stimulus for fastest and slowest 20% of trials
axs[1].text(3, 3, 'Average time\n of stimulus', fontsize=14)
axs[1].vlines(32.65, 0, 2.8, linestyle='solid', color=palette[0])  # -347 ms
axs[1].vlines(12.7, 0, 2.8, linestyle='solid', color=palette[1])  # -746 ms

# Add permutation cluster bars
plt.hlines(-.5, 80, 150, lw=5, alpha=.5, color=palette[0])
plt.hlines(-1, 79, 150, lw=5, alpha=.5, color=palette[1])
plt.hlines(-1.5, 35, 75, lw=5, alpha=.5, color='k')

# Final tweaking
sns.despine(offset=10, bottom=True, trim=True, ax=axs[0])
sns.despine(offset=10, trim=True, ax=axs[1])
plt.tight_layout()

fig.savefig(
    '../pmvig/analysis/manuscript_figs/pmvig_fastest_slowest.tiff', dpi=300)
