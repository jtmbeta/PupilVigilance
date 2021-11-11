#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:14:49 2021

@author: jtm545
"""

# %% Imports
import glob
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from pmvig_func import (new_dataset,
                        read_dataset,
                        mask_blinks,
                        butterworth_series,
                        pupil_preprocessing_fig,
                        extract,
                        reject_bad_trials,
                        mad)

# Sampling rate of the eye tracker
SAMPLE_RATE = 250

# Plot settings
sns.set(context='paper', style='ticks', font='helvetica', font_scale=2.1)

# %% Preprocessing

# Get edf files
edfs = glob.glob('../pmvig/data/**/*.edf')
edfs.sort()

# Lapse thresholds
lapse_thresholds = pd.Series(dtype=float)

# Loop over EDFs
for edf in edfs:
    # Get subject id
    subject = new_dataset(edf)

    # Figure for plotting the processing steps
    fig, axs = pupil_preprocessing_fig(3, subject)

    # Read in samples, blinks and events
    samps, blinks, events = read_dataset(edf)
    events.rename(columns={'event': 'event_type'})

    # Save blinks
    blinks.to_hdf('../pmvig/analysis/blinks.h5', key=subject)
    samps.p.plot(title='Raw', ax=axs[0], legend=False)

    # Mask blinks, interpolate and plot again
    samps = mask_blinks(samps, blinks)
    samps.p.interpolate(inplace=True, limit_direction='both')
    samps.p.plot(title='Masked blinks / interpolated', ax=axs[1], legend=False)

    # Smooth and plot again
    samps = butterworth_series(
        samps,
        fields=['p'],
        filt_order=3,
        cutoff_freq=4/(SAMPLE_RATE/2)  # 4 Hz cutoff
    )
    samps.p.plot(
        title='3rd order Butterworth filter with 4 Hz cut-off',
        ax=axs[2], legend=False
    )

    # Save and close fig
    fig.savefig(f'../pmvig/analysis/processed_pupil_figs/{subject}.png')

    # Save samples
    samps.to_hdf('../pmvig/analysis/samples_250hz.h5', key=subject)

    # Downsample to 50 Hz and save again
    samps = samps[::5]
    samps.to_hdf('../pmvig/analysis/samples_50hz.h5', key=subject)

    # Calculate RTs (time elapsed between stimulus and keypress)
    target = (events.loc[events.event.str.contains('target')]  # Gabor tilts
              .reset_index(drop=True))
    response = (events.loc[events.event.str.contains('response')]  # Keypress
                .reset_index(drop=True))
    response['RT'] = (response.time - target.time)

    # Calculate 1/RT
    response['1/RT'] = (1 / response['RT']) * 1000

    # Lapses are trials where RT is greater than twice the median RT
    medRT = response.RT.median()
    madRT = mad(response.RT)
    lapse_thresh = medRT + (2 * madRT)
    lapse_thresholds = lapse_thresholds.append(
        pd.Series(lapse_thresh, index=[subject]))

    response['Lapse'] = 0
    #response.loc[response.RT > medRT * 2, 'Lapse'] = 1
    response.loc[response.RT > lapse_thresh, 'Lapse'] = 1

    # Group trials into three blocks, and into six groups within a block
    TrialGroup = []
    for t in range(1, 6):
        TrialGroup.append([t] * 18)
    TrialGroup = [tg for sl in TrialGroup for tg in sl]
    response['TrialGroup'] = TrialGroup * 3
    response['Block'] = pd.cut(
        response.time.values, 3, labels=[1, 2, 3]).astype('int')

    # Mark up quintiles for the fastest and slowest 20 percent
    response['Quintile'] = pd.qcut(
        response.RT.values, 5, labels=[1, 2, 3, 4, 5]).astype('int')

    # Add subject ID and make sure we have all the info we need in the response
    # DataFrame
    response['Subject'] = subject
    target = target.merge(
        response[['Block', 'TrialGroup', 'Quintile', 'Subject']],
        left_index=True, right_index=True)
    target.set_index('time', inplace=True)
    response.set_index('time', inplace=True)
    print(f'> Average RT: {int(response.RT.mean())}')

    # Save RTs
    response.to_hdf('../pmvig/analysis/RTs.h5', key=subject)

    # Extract pupil data timelocked to key response
    event_ranges = extract(
        samps,
        response,  # Using the keypress message
        offset=-50,
        duration=150,
        borrow_attributes=['Block', 'TrialGroup', 'Quintile', 'Subject'])

    # Extract baselines (500 ms prestimulus)
    event_baselines = extract(
        samps,
        target,  # Using the target message
        offset=-50,
        duration=50,
        borrow_attributes=['Block', 'TrialGroup', 'Quintile', 'Subject']
    ).groupby(level=0).mean()

    # Calculate baseline z score
    event_baselines['pz'] = ((event_baselines.p
                              - event_baselines.p.mean())
                              / event_baselines.p.std())

    # Divisive correction - we don't want to use this (Mathot, 2018)
    event_ranges['p_div'] = (event_ranges.p / event_baselines.p - 1).values
    
    # Instead we use subtractive correction first and then convert to percent
    # signal change. This approach is less sensetive to distortions that may
    # arise when pupil size at baseline is unrealistically small
    event_ranges['p_sub'] = (event_ranges.p - event_baselines.p).values
    event_ranges['p_pc'] = (event_ranges.p_sub /
                            event_baselines.p * 100).values
    event_ranges['pz'] = (event_ranges.p.sub(
        event_ranges.p.mean())).div(
        event_ranges.p.std())

    # Mark up trials with more than 25 % interpolated data or a blink
    # in the baseline
    event_ranges = reject_bad_trials(event_ranges, interp_thresh=25)
    event_ranges.rename(columns={'reject': 'pct_interp>25'}, inplace=True)
    event_ranges['base_blink'] = event_ranges.index.get_level_values(0).map(
        event_baselines['masked'])

    # Save ranges and baselines
    event_ranges.to_hdf('../pmvig/analysis/event_ranges_50hz.h5', key=subject)
    event_baselines.to_hdf('../pmvig/analysis/event_baselines.h5', key=subject)
