Pupillometry of vigilance
=========================

This repository has the code and instructions for analysing data from the two vigilance experiments described in the manuscript titled "Pupillometry and the vigilance decrement: Task-evoked but not baseline pupil measures reflect declining performance in visual vigilance tasks". 

Set up environment
------------------

Install Anaconda and then run the following commands in a terminal or Anaconda Prompt:

```
conda create --name vigilance python=3.7.7
```

```
conda activate vigilance
```

```
conda install anaconda
```

```
conda install -c conda-forge mne
```

Obtain raw data from figshare
-----------------------------

https://doi.org/10.6084/m9.figshare.17317886.v1

Once obtained, unzip the data and place in the correct folders.

Review the scripts
------------------

You may need to adjust paths and folder names to account for your local directory structure.

Experiment 1 (vigilance)
------------------------

1. Run vig_preproc.py
2. Run vig_postproc.py

Experiment 2 (psychomotor vigilance)
------------------------------------

1. Run pmvig_preproc.py
2. Run pmvig_postproc.py
