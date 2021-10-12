# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob

import pandas as pd

from vig_func import (new_dataset,
                      read_dataset, 
                      mask_blinks,
                      butterworth_series, 
                      calculate_RTs,
                      calculate_sd_outcomes,
                      pupil_preprocessing_fig,
                      extract)


edfs = glob.glob('../vig/data/**/*.edf')
edfs.sort()
edfs = [edf for edf in edfs if '027' not in edf]
edfs = edfs[1:2]
for edf in edfs:
    subject = new_dataset(edf)
    fig, axs = pupil_preprocessing_fig(3, edf)
    samps, blinks, events = read_dataset(edf)
    blinks.to_hdf('../vig/data/blinks.h5', key=subject)
    samps.p.plot(title='Raw', ax=axs[0], legend=True)
    samps = mask_blinks(samps, blinks)
    samps.p.interpolate(inplace=True, limit_direction='both')
    samps.p.plot(title='Masked blinks / interpolated', ax=axs[1], legend=False)
    samps = butterworth_series(
        samps,
        fields=['p'],
        filt_order=3,
        cutoff_freq=4/(250/2))
    samps.p.plot(
        title='3rd order Butterworth filter with 4 Hz cut-off', ax=axs[2], legend=False)
    samps.to_hdf('../vig/analysis/samples_250hz.h5', key=subject)
    samps = samps[::5]
    samps.to_hdf('../vig/analysis/samples_50hz.h5', key=subject)
    # events
    events = calculate_RTs(events)
    events = calculate_sd_outcomes(events)
    events.to_hdf('../vig/analysis/events.h5', key=subject)
    
    # extract ranges for all normal and rare events
    event_ranges = extract(samps, 
                           events.loc[events.event!='key'][:-1], # exclude last event
                           offset=-25, 
                           duration=125,
                           borrow_attributes=['event','outcome'])
    
    # baseline correction
    event_baselines = event_ranges.loc[:, slice(0,25), :].median(level=0)
    event_ranges['p_div'] = (event_ranges.p / event_baselines.p - 1).values
    event_ranges['p_sub'] = (event_ranges.p - event_baselines.p).values
    event_ranges.to_hdf('../vig/analysis/event_ranges_50hz.h5', key=subject)

    # extract ranges for all key events
    key_ranges = extract(samps, 
                         events.loc[events.event=='key'], 
                         offset=-50, 
                         duration=125,
                         borrow_attributes=['event','outcome'])
    # baseline correction
    # baseline taken between -1000:-500 ms prior to response (med_RT = 660 ms)
    key_baselines = key_ranges.loc[:, slice(0,25), :].median(level=0)
    key_ranges['p_div'] = (key_ranges.p / key_baselines.p - 1).values
    key_ranges['p_sub'] = (key_ranges.p - key_baselines.p).values
    key_ranges.to_hdf('../vig/analysis/key_ranges_50hz.h5', key=subject)


    #new_onset = (df.index.get_level_values('onset').unique() - abs(offset)) / sample_rate
    #df.index = df.index.set_levels(levels=new_onset, level='onset')
    
