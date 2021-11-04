#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:21:31 2021

@author: jtm545
"""

# %% Imports  and config

import glob
import sys

from scipy.stats import norm
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
import pandas as pd
import seaborn as sns

from vig_func import (new_dataset,
                      read_dataset,
                      mask_blinks,
                      butterworth_series,
                      calculate_RTs,
                      calculate_sd_outcomes,
                      pupil_preprocessing_fig,
                      extract,
                      reject_bad_trials,
                      mad)

sns.set(context='paper', style='ticks', font='helvetica', font_scale=2.1)

# %%  Plot histogram of RTs (hits and false alarms)

# Open the store and filter the metadata
key_store = pd.HDFStore('../analysis/key_events.h5')
keys = [k for k in key_store.keys() if not 'meta' in k]

# Gather key presses from all subjects into a single DataFrame
df = pd.DataFrame()
for ks in key_store:
    d = key_store[ks]
    d['subject'] = ks[1:]
    df = df.append(d)
df = df.reset_index(drop=True)
key_store.close()

# Save descriptive statistivcs of RTs for Subject | Outcome
df.groupby(by=['subject', 'outcome'])['RT'].describe().to_csv(
    '../analysis/RT_descriptive_statistics.csv')

# Median and median absolute deviation RT for hits across all subjects. We only
# consider RTs less than 6000 ms, which is the minimum time between rare events
med_RT = df.loc[df.RT <= 6000, 'RT'].median()
mad_RT = mad(df.loc[df.RT <= 6000, 'RT'])

# Get the RTs, but not the rush trials
rts = df.loc[df.RT != -1, 'RT']

# Two sets of bins with different widths to show hits and false alarms clearly
bins = [b for b in range(0, 6000, 50)]
bins2 = [b for b in range(2000, 18000, 500)]

# Plot the histograms
fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
sns.histplot(rts, ax=ax, bins=bins, kde=False, color='k')
sns.histplot(rts, ax=ax2, bins=bins2, kde=False, color='k')

# Tweak aesthetics
# 0 - 2000 ms RTs
ax.set(xlim=(0, 2000),
       ylim=(0, 600),
       ylabel='Frequency',
       xlabel='')
ax.axvspan(120, 1200, color='green', alpha=.1)
ax.axvspan(0, 120, color='red', alpha=.1)
ax.axvspan(1200, 15000, color='red', alpha=.1)
ax.set_title('Bin width = 50 ms', size=15)
plt.setp(ax.get_xticklabels(), ha='right', rotation=45)

# 2000 - 18000 ms RTs
ax2.yaxis.tick_right()
ax2.set(xlim=(2000, 18000),
        xticks=[2000, 10000, 18000],
        ylim=(0, 600),
        ylabel='Frequency',
        xlabel='')
ax2.axvspan(1000, 1200, color='green', alpha=.1)
ax2.axvspan(1200, 18000, color='red', alpha=.1)
ax2.set_title('Bin width = 500 ms', size=15)
plt.setp(ax2.get_xticklabels(), ha='right', rotation=45)

fig.tight_layout()
fig.text(.55, .000001, 'Time to last critical signal (ms)', ha='center')
fig.savefig(
    '../analysis/manuscript_figs/rtsdist.tiff', dpi=300, bbox_inches='tight')

# %% Calculate percentage outcomes (H, M, CR, FA) per Subject | Watch period

# Open the store and filter the metadata
event_store = pd.HDFStore('../analysis/exp_events.h5')
keys = [k for k in event_store.keys() if not 'meta' in k]

# Gather events from all subjects into a single DataFrame
df = pd.DataFrame()
for k in keys:
    d = event_store[k]
    d['subject'] = k[1:]
    df = df.append(d)
df = df.reset_index(drop=True)
event_store.close()

# New target variable
df['target'] = 0
df.loc[df.event == 'rare', 'target'] = 1

# Calculate percentages
accuracy = (df.groupby(['subject', 'timeblock', 'target', 'outcome'])
            .agg({'event': 'count'})
            .reset_index()
            .pivot_table(index='subject',
                         values='event',
                         columns=['timeblock', 'target', 'outcome'])
            .fillna(0)
            .unstack()
            .rename('proportion', axis=0)
            .to_frame()
            .reorder_levels(['subject', 'timeblock', 'target', 'outcome'])
            .groupby(level=[0, 1, 2])
            .apply(lambda x: (x + .5) / float(x.sum() + 1)))  # Loglinear thing
accuracy['percentage'] = accuracy.proportion * 100.
accuracy = accuracy.reset_index()

# Save for repeated measures analysis in JASP / SPSS
accuracy_rm = accuracy.pivot(
    index='subject', columns=['timeblock', 'outcome'], values=['percentage'])
accuracy_rm.columns = accuracy_rm.columns.map('|'.join).str.strip('|')
accuracy_rm.to_csv('../analysis/rm_percentage_outcomes.csv')

# %% Calculate signal detection outcomes (d' and c) per Subject | Watch period

sdt = accuracy.pivot_table(index=['subject', 'timeblock'],
                           values='proportion',
                           columns=['outcome'])
sdt['dp'] = norm.ppf(sdt.loc[:, 'H']) - norm.ppf(sdt.loc[:, 'FA'])
sdt['c'] = -(norm.ppf(sdt.loc[:, 'H']) + norm.ppf(sdt.loc[:, 'FA'])) / 2
sdt = sdt.reset_index()

# Save for repeated measures analysis in JASP / SPSS
sdt_rm = sdt.pivot(index='subject', columns=['timeblock'], values=['dp', 'c'])
sdt_rm.columns = sdt_rm.columns.map('|'.join).str.strip('|')
sdt_rm.to_csv('../analysis/rm_c_dp_timeblock.csv')

# %% Plot performance measures

fig = plt.figure(figsize=(12, 7))

# Plot percentage of hits
ax1 = fig.add_subplot(231)
sns.pointplot(x='timeblock', y='percentage',
              data=accuracy.loc[accuracy.outcome == 'H'],
              color='k', markers='s', ax=ax1)
ax1.set_xlabel('Watch Period')
ax1.set_ylabel('Hits (%)')
ax1.set_ylim((45, 85))

# Plot percentage of false alarms
ax2 = fig.add_subplot(232)
sns.pointplot(x='timeblock', y='percentage',
              data=accuracy.loc[accuracy.outcome == 'FA'],
              color='k', markers='s', ax=ax2)
ax2.set_xlabel('Watch Period')
ax2.set_ylabel('False alarms (%)')
ax2.set_ylim((0, 3))

# Plot RT for hits
rt_hits = df.loc[df.outcome == 'H'].groupby(
    ['subject', 'timeblock'], as_index=False)['RT'].mean()
ax3 = fig.add_subplot(233)
sns.pointplot(x='timeblock', y='RT',
              data=rt_hits,
              color='k', markers='s', ax=ax3)
ax3.set_xlabel('Watch Period')
ax3.set_ylabel('RT (ms)')
ax3.text(1, 630, '(Hits)')
ax3.set_ylim((600, 800))

# Plot d'
ax4 = fig.add_subplot(234)
sns.pointplot(x='timeblock', y='dp',
              data=sdt,
              color='k', markers='s', ax=ax4)
ax4.set_xlabel('Watch Period')
ax4.set_ylabel("Sensitivity ($d'$)")
ax4.set_ylim((2, 4))

# Plot c
ax5 = fig.add_subplot(235)
sns.pointplot(x='timeblock', y='c',
              data=sdt,
              color='k', markers='s', ax=ax5)
ax5.set_xlabel('Watch Period')
ax5.set_ylabel("Response bias ($c$)")
ax5.set_ylim((.7, 1.4))

plt.tight_layout()

# Save figure, also RT hit data for repeated measures analysis in JASP / SPSS
fig.savefig('../analysis/manuscript_figs/behavioral_data.tiff', dpi=300)
rthits_rm = rt_hits.pivot(index='subject', columns=['timeblock'], values='RT')
rthits_rm.columns = rthits_rm.columns.map('|'.join).str.strip('|')
rthits_rm.to_csv('../analysis/rm_rt_hits_timeblock.csv')

# %% Plot response-locked (H, FA) pupil traces

# Open the store
key_ranges = pd.HDFStore('../analysis/key_ranges_50hz.h5')
key_baselines = pd.HDFStore('../analysis/baselines_')

# Gather data from all subjects into a single DataFrame
df = pd.DataFrame()
for k in key_ranges.keys():
    d = key_ranges[k].rename(columns={'event': 'event_type'}).reset_index()
    d = d[d.reject == 0]
    d['subject'] = k[1:]
    d['pz'] = (d.p - d.p.mean()) / d.p.std()
    d = d.groupby(by=['subject', 'outcome', 'onset'], as_index=False).mean()
    d.onset = (d.onset - 50) * (1/50)
    df = df.append(d)
key_ranges.close()

# Filter hits and false alarms, aggregate for permutation tests
df = df[df.outcome.isin(['H', 'FA'])]
df = df.groupby(['subject', 'outcome', 'onset']).mean()

# Perform cluster-based permutation stats
print('*** Cluster permutation tests for response-locked pupil traces ***\n')
data = df['p_pc'].unstack()
result_H = permutation_cluster_1samp_test(
    data.loc[:, 'H', :].values, out_type='mask')
print('Slices and p-values ')
print(result_H[1:3])
result_FA = permutation_cluster_1samp_test(
    data.loc[:, 'FA', :].values, out_type='mask')
print(result_FA[1:3])
diff = data.loc[:, 'H', :].values - data.loc[:, 'FA', :].values
result_diff = permutation_cluster_1samp_test(diff, out_type='mask')
print(result_diff[1:3])

# Colours for hits and false alarms
palette = sns.color_palette()
color_mapping = {'H': palette[4], 'FA': palette[3]}

# Plot the pupil traces
fig, ax = plt.subplots()
ax = sns.lineplot(data=df, x='onset', y='p_sub', 
                  hue='outcome', ci=68,
                  estimator='mean',  # lambda x: sum(x)*.1/len(x)
                  ax=ax, palette=color_mapping,
                  n_boot=5000)

# Lines for stim onset and baseline
ax.vlines(0, ax.get_ybound()[0], ax.get_ybound()[1], 
          linestyles='dashed', color='k')
ax.hlines(0, -1, 1.5, linestyles='dashed', color='k')
ax.set(xlabel='Time (s)',
       ylabel='Pupil modulation (%)',
       title='Button-locked',
       ylim=(-4,12))
ln1 = ax.hlines(-.5, (65-50)/50, (97-50)/50, lw=5, alpha=.5,color=p[4])
ln2 = ax.hlines(-1, (36-50)/50, (104-50)/50, lw=5, alpha=.5,color=p[3])
ax.text(.5,-3,'$P$ < .05', {'size':17})
lgd = ax.get_legend()
lgd.set_title('')
sns.despine(offset=10, trim=True)
fig.savefig('../analysis/button_FA_H_pupil.svg')


# testing ground

#data = df['p_pc'].unstack().groupby('outcome').mean().T.plot()
#data = df['p_sub'].unstack().groupby('outcome').mean().T.plot()
