#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:21:31 2021

@author: jtm545
"""

# %% Imports  and config

from scipy.stats import norm
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
import pandas as pd
import seaborn as sns
import numpy as np

from vig_func import mad

sns.set(context='paper', style='ticks', font='helvetica', font_scale=2.1)

# %%  Plot histogram of RTs (hits and false alarms)

# Open the store and filter the metadata
key_store = pd.HDFStore('../vig/analysis/key_events.h5')

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
    '../vig/analysis/RT_descriptive_statistics.csv')

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
    '../vig/analysis/manuscript_figs/rtsdist.tiff',
    dpi=300, bbox_inches='tight')

# %% Calculate percentage outcomes (H, M, CR, FA) per Subject | Watch period

# Open the store and filter the metadata
event_store = pd.HDFStore('../vig/analysis/exp_events.h5')
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
accuracy_rm.to_csv('../vig/analysis/rm_percentage_outcomes.csv')

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
sdt_rm.to_csv('../vig/analysis/rm_c_dp_timeblock.csv')

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
rt_fa = df.loc[df.outcome == 'FA'].groupby(
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
fig.savefig('../vig/analysis/manuscript_figs/behavioral_data.tiff', dpi=300)
rthits_rm = rt_hits.pivot(index='subject', columns=['timeblock'], values='RT')
rthits_rm.columns = rthits_rm.columns.map('|'.join).str.strip('|')
rthits_rm.to_csv('../vig/analysis/rm_rt_hits_timeblock.csv')

# %% Cluster permutation stats for button-locked (H, FA) pupil traces

# Open the store
key_ranges = pd.HDFStore('../vig/analysis/key_ranges_50hz.h5')
key_baselines = pd.HDFStore('../vig/analysis/key_baselines_50hz.h5')

# Gather data from all subjects into a single DataFrame
df_keys = pd.DataFrame()
df_key_baselines = pd.DataFrame()
df_key_evoked_scalar = pd.DataFrame()
for k in key_ranges.keys():

    # Load pupil data and add subject ID
    p = key_ranges[k].rename(columns={'event': 'event_type'}).reset_index()
    p['subject'] = k[1:]

    # Load baselines and drop those which contain a blink
    bls = key_baselines[k].reset_index()
    bls['subject'] = k[1:]
    categories = (p.groupby(['event', 'outcome', 'timeblock'], as_index=False)
                   .count()[['event', 'outcome', 'timeblock']])
    bls = bls.merge(categories, on='event')

    # Get indices of valid trials. These are trials which contain less than
    # 25 percent interpolated data and which do not contain a blink in the
    # baseline period.
    use_trials = np.intersect1d(
        p[p.reject == 0].event.unique(),  # As marked by reject_bad_trials
        bls[bls.masked == 0].event.to_numpy())  # No blink in baseline

    # Calculate z score of pupil data
    p['pz'] = (p.p - p.p.mean()) / p.p.std()
    bls['pz'] = (bls.p - bls.p.mean()) / bls.p.std()

    # Get the scalar values for evoked responses
    evoked = (p.loc[((p.event.isin(use_trials)) & (p.onset >= 50))]
              .groupby(by=['subject', 'timeblock', 'outcome'], as_index=False)
              .mean())

    # Drop trials with blink in baseline or too much interpolated data and
    # then average by subject / outcome / onset
    p = (p.set_index('event').loc[use_trials]
         .reset_index()
         .groupby(by=['subject', 'outcome', 'onset'], as_index=False)
         .mean())
    bls = (bls.set_index('event').loc[use_trials]
           .reset_index()
           .groupby(by=['subject', 'timeblock', 'outcome'],
                    as_index=False)['pz']
           .mean())

    # Time relevant to stimulus onset. For button events, the baseline is from
    # -1000:-500 ms.
    p['Time'] = (p.onset - 50) * (1/50)
    # Add to aggregated DFs
    df_keys = df_keys.append(p)
    df_key_baselines = df_key_baselines.append(bls)
    df_key_evoked_scalar = df_key_evoked_scalar.append(evoked)

# Close the stores
key_ranges.close()
key_baselines.close()

df_keys.to_csv(
    '../vig/analysis/ag_pupil_key_traces_subject_outcome_onset.csv')
df_key_baselines.to_csv(
    '../vig/analysis/ag_pupil_key_baselines_subject_outcome_onset.csv')
df_key_evoked_scalar.to_csv(
    '../vig/analysis/ag_pupil_event_evoked_scalar_subject_outcome_onset.csv')

# Set relevant multi index
df_keys = df_keys.set_index(['subject', 'outcome', 'onset'])

# Perform cluster-based permutation stats
print('*** Cluster permutation tests for response-locked pupil traces ***\n')

# Prepare the data
data = df_keys['p_pc'].unstack()

# Test for hits
print('\n> Hits')
result_H = permutation_cluster_1samp_test(
    data.loc[:, 'H', :].values, out_type='mask')
print(f'> Significance and slice indices: {result_H[1:3]}')

# Test for false alarms
print('\n> False alarms')
result_FA = permutation_cluster_1samp_test(
    data.loc[:, 'FA', :].values, out_type='mask')
print(f'> Significance and slice indices: {result_FA[1:3]}')

# Test for difference
print('\n> Hits minus false alarms (difference)')
diff = data.loc[:, 'H', :].values - data.loc[:, 'FA', :].values
result_diff_H_FA = permutation_cluster_1samp_test(diff, out_type='mask')
print(f'> Significance and slice indices: {result_diff_H_FA[1:3]}')

# %% Cluster permutation stats for event-locked (M, CR) pupil traces

# Open the store
event_ranges = pd.HDFStore('../vig/analysis/event_ranges_50hz.h5')
event_baselines = pd.HDFStore('../vig/analysis/event_baselines_50hz.h5')

# Gather data from all subjects into single DataFrames
df_events = pd.DataFrame()
df_event_baselines = pd.DataFrame()
df_event_evoked_scalar = pd.DataFrame()
for k in event_ranges.keys():

    # Load pupil data, keep only misses and correct rejections, add subject ID
    p = event_ranges[k].rename(columns={'event': 'event_type'}).reset_index()
    p = p.loc[p.outcome.isin(['CR', 'M'])]
    p['subject'] = k[1:]

    # As above but for baselines
    bls = event_baselines[k].reset_index()
    bls['subject'] = k[1:]
    categories = (p.groupby(['event', 'outcome', 'timeblock'], as_index=False)
                   .count()[['event', 'outcome', 'timeblock']])
    bls = bls.merge(categories, on='event').reset_index()
    bls = bls.loc[bls.outcome.isin(['CR', 'M'])]

    # Get indices of valid trials. These are trials which contain less than
    # 25 percent interpolated data and which do not contain a blink in the
    # baseline period.
    use_trials = np.intersect1d(
        p[p.reject == 0].event.unique(),  # As marked by reject_bad_trials
        bls[bls.masked == 0].event.to_numpy())  # No blink in baseline

    # Calculate z score of pupil data for events and baselines
    p['pz'] = (p.p - p.p.mean()) / p.p.std()
    bls['pz'] = (bls.p - bls.p.mean()) / bls.p.std()

    # Get the scalar values for evoked responses
    evoked = (p.loc[((p.event.isin(use_trials)) & (p.onset >= 25))]
              .groupby(by=['subject', 'timeblock', 'outcome'], as_index=False)
              .mean())

    # Drop trials with blink in baseline or too much interpolated data and
    # then average by subject / outcome / onset
    p = (p.set_index('event').loc[use_trials]
         .reset_index()
         .groupby(by=['subject', 'outcome', 'onset'], as_index=False)
         .mean())
    bls = (bls.set_index('event').loc[use_trials]
           .reset_index()
           .groupby(by=['subject', 'timeblock', 'outcome'],
                    as_index=False)['pz']
           .mean())

    # Time relevant to stimulus onset (500 ms baseline for events)
    p['Time'] = (p.onset - 25) * (1/50)

    # Add to aggregated DFs
    df_events = df_events.append(p)
    df_event_baselines = df_event_baselines.append(bls)
    df_event_evoked_scalar = df_event_evoked_scalar.append(evoked)

# Close the stores
event_ranges.close()
event_baselines.close()

# Save to CSV format
df_events.to_csv(
    '../vig/analysis/ag_pupil_event_traces_subject_outcome_onset.csv')
df_event_baselines.to_csv(
    '../vig/analysis/ag_pupil_event_baselines_subject_outcome_onset.csv')
df_event_evoked_scalar.to_csv(
    '../vig/analysis/ag_pupil_event_evoked_scalar_subject_outcome_onset.csv')

# Set relevant multi index
df_events = df_events.set_index(['subject', 'outcome', 'onset'])

# Perform cluster-based permutation stats
print('*** Cluster permutation tests for event-locked pupil traces ***\n')

# Prepare the data
data = df_events['p_pc'].unstack()

# Test for hits
print('\n> Correct rejections')
result_CR = permutation_cluster_1samp_test(
    data.loc[:, 'CR', :].values, out_type='mask')
print(f'> Significance and slice indices: {result_CR[1:3]}')

# Test for false alarms
print('\n> Misses')
result_M = permutation_cluster_1samp_test(
    data.loc[:, 'M', :].values, out_type='mask')
print(f'> Significance and slice indices: {result_M[1:3]}')

# Test for difference
print('\n> Correct rejections minus misses (difference)')
diff = data.loc[:, 'CR', :].values - data.loc[:, 'M', :].values
result_diff_CR_M = permutation_cluster_1samp_test(diff, out_type='mask')
print(f'> Significance and slice indices: {result_diff_CR_M[1:3]}')

# %% Plot the pupil data for stimulus and button events

# Colour mappings
palette = sns.color_palette()
color_mapping_CR_M = {'CR': palette[0], 'M': palette[2]}
color_mapping_H_FA = {'H': palette[4], 'FA': palette[3]}

# Plot the pupil traces
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Correct rejections and misses first
sns.lineplot(data=df_events, x='onset', y='p_pc',
             hue='outcome', ci=68,  estimator='mean',
             ax=axs[0], palette=color_mapping_CR_M,
             n_boot=5000)

# Lines for stim onset and baseline
axs[0].set_title('Event-locked')
axs[0].axvline(25, 0, 1, linestyle='dashed', color='k')
axs[0].set_xticks(np.arange(0, 150, 25))
axs[0].set_xticklabels([str(val) for val in np.arange(-.5, 2.5, .5)])


# Coloured bars to show significant clusters. The extent of these bars is
# determined by the slice indices of the significant permutation tests
# performed in the previous cells
axs[0].hlines(-1, 51, 125, lw=5, alpha=.5, color=palette[2])
axs[0].hlines(-1.5, 50, 125, lw=5, alpha=.5, color='k')
axs[0].text(60, -2.7, '$P$ < .05', {'size': 17})

# Now plot hits and false alarms
sns.lineplot(data=df_keys, x='onset', y='p_pc',
             hue='outcome', ci=68, estimator='mean',
             ax=axs[1], palette=color_mapping_H_FA,
             n_boot=5000)

# Lines for stim onset and baseline
axs[1].set_title('Button-locked')
axs[1].axvline(50, 0, 1, linestyle='dashed', color='k')
axs[1].set_xticks(np.arange(0, 150, 25))
axs[1].set_xticklabels([str(val) for val in np.arange(-1., 2, .5)])

# Coloured bars to show significant clusters. The extent of these bars is
# determined by the slice indices of the significant permutation tests
axs[1].hlines(-.5, 41, 105, lw=5, alpha=.5, color=palette[4])
axs[1].hlines(-1, 37, 125, lw=5, alpha=.5, color=palette[3])
axs[1].hlines(-1.5, 91, 125, lw=5, alpha=.5, color='k')
axs[1].text(60, -2.7, '$P$ < .05', {'size': 17})

# General tweaking for both subplots
for ax in axs:
    ax.get_legend().set_title('')
    ax.set(ylim=(-4, 10),
           xlabel='Time (s)',
           ylabel='Pupil modulation (%)')
    ax.axhline(0, 0, 1, linestyle='dashed', color='k')

# Finish up
sns.despine(offset=10, trim=True)
plt.tight_layout(h_pad=3)

# Save figure
fig.savefig('../vig/analysis/event_button_pupil.tiff', dpi=300)

# %% Plot the scalar responses

# # Figure
fig, axs = plt.subplots(2, 2, figsize=(8, 7))
axs = axs.flatten()

# Plot baseline M/CR
sns.pointplot(
    data=df_event_baselines, x='timeblock', y='pz',  # z score
    hue='outcome', dodge=.3, markers='s', ci=95, units='subject',
    ax=axs[0], palette=color_mapping_CR_M
)
axs[0].set(
    ylim=(-.6, .6),
    ylabel='Pupil size (z)'
)
axs[0].legend([], [], frameon=False)

# Plot button event M/CR
sns.pointplot(
    data=df_event_evoked_scalar.reset_index(), x='timeblock', y='p_pc',
    hue='outcome', dodge=.3, markers='s', ci=95, units='subject',
    ax=axs[1], palette=color_mapping_CR_M
)
axs[1].set(
    ylim=(-2, 15),
    ylabel='Pupil modulation (%)'
)
axs[1].get_legend().set_title('')

# Hits / False Alarms
# Plot baseline H/FA
sns.pointplot(
    data=df_key_baselines, x='timeblock', y='pz',
    hue='outcome', dodge=.3, markers='s', ci=95, units='subject',
    ax=axs[2], palette=color_mapping_H_FA
)
axs[2].set(
    ylim=(-.6, .6),
    ylabel='Pupil size (z)'
)
axs[2].legend([], [], frameon=False)

# Plot button event H/FA
sns.pointplot(
    data=df_key_evoked_scalar, x='timeblock', y='p_pc',
    hue='outcome', dodge=.3, markers='s', ci=95, units='subject',
    ax=axs[3], palette=color_mapping_H_FA
)
axs[3].set(
    ylim=(-2, 15),
    ylabel='Pupil modulation (%)'
)
axs[3].get_legend().set_title('')

# General subplot tweaking
for ax in axs:
    ax.set(xlabel='Watch Period')
plt.tight_layout()

# Save the figure
fig.savefig('../vig/analysis/manuscript_figs/scalar_pupil.tiff',
            dpi=300, bbox_inches='tight')

# %% Save for repeated measures analysis in SPSS / JASP

# Event baselines
df1 = df_event_baselines.pivot(
    index='subject', columns=['timeblock', 'outcome'], values='pz')
cols = df1.columns.map('|'.join).str.strip('|')
df1.columns = ['event_base|' + val for val in cols]

# Key baselines
df2 = df_key_baselines.pivot(
    index='subject', columns=['timeblock', 'outcome'], values='pz')
cols = df2.columns.map('|'.join).str.strip('|')
df2.columns = ['key_base|' + val for val in cols]

# Event evoked
df3 = df_event_evoked_scalar.pivot(
    index='subject', columns=['timeblock', 'outcome'], values='p_pc')
cols = df3.columns.map('|'.join).str.strip('|')
df3.columns = ['event_evoked|' + val for val in cols]

# Key evoked
df4 = df_key_evoked_scalar.pivot(
    index='subject', columns=['timeblock', 'outcome'], values='p_pc')
cols = df4.columns.map('|'.join).str.strip('|')
df4.columns = ['key_evoked|' + val for val in cols]

# Save as single DataFrame
rm_pupil = pd.concat([df1, df2, df3, df4], axis=1)
rm_pupil.to_csv('../vig/analysis/rm_pupil.csv')
