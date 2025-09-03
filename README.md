# PSRJ1713-0747-Profile-Change
These scripts were used to process and analyse the April 2021 profile change event that occurred on the millisecond pulsar (MSP) PSR J1713+0747. The results from this analysis can be found in Mandow et al. 2025 <link>

The raw data for this analysis has already been flux and polarization calibrated, and sub-banded. Additional treatement of the data files includes de-dispersing to the dispersion measure value and then correcting for rotation measure.

Low S/N observations (where threshold < 50) were discarded.

The timeseries analysis script will generate individual profile residuals (as text files) for each epoch per Stokes I, linear polarisation and Stokes V. These individual epochs are then placed into a 2D numpy array so that they become a timeseries of profile residuals.

## Purpose
The purpose of this code is to allow each epoch observation to be compared to the stable template for each sub-band. Differences between the epoch observation (sample) and the template are known as profile residuals. Where no profile shape change has occurred then the profile residuals will reflect Guassian noise across the pulse phase. Where a profile residual occurs in a certain bin, it will appear as a significant deviation away from the baseline noise in either a positive or negative value. 
