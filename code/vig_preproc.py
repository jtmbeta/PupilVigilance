#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preprocessing script for vigilance experiment
=============================================

@author: jtm

"""

# %% Imports  and config

import glob

import seaborn as sns
import pandas as pd

from vig_func import (new_dataset,
                      read_dataset,
                      mask_blinks,
                      butterworth_series,
                      calculate_RTs,
                      calculate_sd_outcomes,
                      pupil_preprocessing_fig,
                      extract,
                      reject_bad_trials)

# Sampling rate of the eye tracker
SAMPLE_RATE = 250

# Plot settings
sns.set(context='paper', style='ticks', font='helvetica', font_scale=2.1)

# %% Main preprocessing loop

# Get the EDFs
edfs = glob.glob('../vig/data/**/*.edf')
edfs.sort()

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
    blinks.to_hdf('../vig/analysis/blinks.h5', key=subject)
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
    fig.savefig(f'../vig/analysis/processed_pupil_figs/{subject}.png')

    # Save samples
    samps.to_hdf('../vig/analysis/samples_250hz.h5', key=subject)

    # Downsample to 50 Hz and save again
    samps = samps[::5]
    samps.to_hdf('../vig/analysis/samples_50hz.h5', key=subject)

    # Caluclate RTs and signal detection theory outcomes
    events = calculate_RTs(events)
    events = calculate_sd_outcomes(events)

    # Categorize by time periods (i.e., periods of watch)
    events['timeblock'] = pd.cut(
        events.time, bins=3, labels=[1, 2, 3]).astype('str')
    events['min'] = pd.cut(
        events.time, bins=30, labels=range(0, 30)).astype('str')

    # Separate key responses and experimental events, then save
    exp_events = events.loc[events.event_type != 'key']
    key_events = events.loc[events.event_type == 'key']
    exp_events.to_hdf('../vig/analysis/exp_events.h5',
                      key=subject, format='table',
                      data_columns=True, append=False)
    key_events.to_hdf('../vig/analysis/key_events.h5',
                      key=subject, format='table',
                      data_columns=True, append=False)

    # Extract ranges for all normal and rare events
    event_ranges = extract(samps,
                           exp_events[:-1],  # exclude last event to fix bug
                           offset=-25,  # 500 ms before button press (baseline)
                           duration=125,  # 2.5 seconds of data
                           borrow_attributes=['event_type',
                                              'outcome',
                                              'timeblock',
                                              'min'])
    # Baseline correction
    # Get the baselines
    event_baselines = event_ranges.loc[:, slice(0, 25), :].groupby(
        level=0).mean()

    # Divisive correction - we don't want to use this (Mathot, 2018)
    event_ranges['p_div'] = (event_ranges.p / event_baselines.p - 1).values

    # Instead we use subtractive correction first and then convert to percent
    # signal change. This approach is less sensetive to distortions that may
    # arise when pupil size at baseline is unrealistically small
    event_ranges['p_sub'] = ((event_ranges.p
                             - event_baselines.p).values)
    event_ranges['p_pc'] = ((event_ranges.p_sub
                            / event_baselines.p * 100).values)

    # Markup trials with more than 25 percent interpolated data
    event_ranges = reject_bad_trials(
        event_ranges, interp_thresh=25, drop=False)

    # Save events
    event_ranges.to_hdf('../vig/analysis/event_ranges_50hz.h5', key=subject)
    event_baselines.to_hdf(
        '../vig/analysis/event_baselines_50hz.h5', key=subject)

    # Extract ranges for all key events (i.e., button presses)
    key_ranges = extract(samps,
                         key_events,
                         offset=-50,  # 1 second before button press
                         duration=125,  # 2.5 seconds of data
                         borrow_attributes=['event_type',
                                            'outcome',
                                            'timeblock',
                                            'min',
                                            'event'])
    
    key_ranges = key_ranges.rename(columns={'event': 'stim_event'})
    
    # Baseline correction
    # Baseline taken between -1000:-500 ms prior to response (med_RT = 660 ms)
    key_baselines = key_ranges.loc[:, slice(0, 25), :].groupby(
        level=0).mean()

    # As above
    key_ranges['p_div'] = ((key_ranges.p
                            / key_baselines.p - 1).values)
    key_ranges['p_sub'] = ((key_ranges.p
                            - key_baselines.p).values)
    key_ranges['p_pc'] = ((key_ranges.p_sub
                           / key_baselines.p * 100).values)

    # Exclude trials with more than 25 percent interpolated data
    key_ranges = reject_bad_trials(key_ranges, interp_thresh=25, drop=False)

    # Save key events
    key_ranges.to_hdf('../vig/analysis/key_ranges_50hz.h5', key=subject)
    key_baselines.to_hdf('../vig/analysis/key_baselines_50hz.h5', key=subject)
    