# PyITD Intrinsic Time-Scale Decomposition
## Introduction

This is a Python implementation of Intrinsic Time-Scale Decomposition (ITD).
ITD is a patented decomposition method developed under grant.
NIH/NINDS grants nos. 1R01NS046602-01 and 1R43NS39240-01.
This algorithm is patented US-7457756-B1 by Frei And Osorio, 2027-02-16 Adjusted expiration

"We introduce a new algorithm, the intrinsic time-scale decomposition (ITD), for efficient
and precise time–frequency–energy (TFE) analysis of signals. The ITD method
overcomes many of the limitations of both classical (e.g. Fourier transform or wavelet
transform based) and more recent (empirical mode decomposition based) approaches to
TFE analysis of signals that are nonlinear and/or non-stationary in nature. The ITD
method decomposes a signal into (i) a sum of proper rotation components, for which
instantaneous frequency and amplitude are well defined, and (ii) a monotonic trend. The
decomposition preserves precise temporal information regarding signal critical points
and riding waves, with a temporal resolution equal to the time-scale of extrema
occurrence in the input signal. We also demonstrate how the ITD enables application of
single-wave analysis and how this, in turn, leads to a powerful new class of real-time
signal filters, which extract and utilize the inherent instantaneous amplitude and
frequency/phase information in combination with other relevant morphological features."
Authors: Mark Frei, Ivan Osorio (2007).

The Intrinsic Time-Scale Decomposition (ITD) is a purely algorithmic, non-lossy
iterative decomposition of a time series {Y (i)}N i=1. At the first stage, the signal is
decomposed into a proper rotation R1(i), an oscillating mode in which maxima and
minima are positive and negative, respectively, and a residual B1(i) called baseline .
The baseline B1 is now decomposed in the same fashion, producing a proper rotation
R2 and a baseline B2 , and so on. The process stops when the resulting baseline has
only two extrema, or is a constant.

B2 = B1 - R1, and so on. The decomposition mode is fully reversable with typically perfect reconstitution.
The final output for the decomposition is a monotonic upward trend.

ITD avoids a priori assumptions about the content/morphology of the signal
being analysed (e.g. make the decomposition ‘basis free’)
ITD also performs the analysis in an efficient and rapid manner with O(n) computations

 ITD provides:
 efficient signal decomposition into ‘proper rotation’ components, for which
instantaneous frequency and amplitude are well defined, along with the
underlying monotonic signal trend, without the need for laborious and
ineffective sifting or splines,
precise temporal information regarding instantaneous frequency and
amplitude of component signals with a temporal resolution equal to the
time-scale of occurrence of extrema in the input signal, 
a new class of real-time signal filters that utilize the newly available
instantaneous amplitude and frequency/phase information together with
additional features and morphology information obtained via single-wave
analysis. Moreover, the resulting feature extraction and feature-based
filtering can be performed in real-time and is easily adapted to extract
feature signals of interest at the time-scales on which they naturally occur,
while preserving their morphology and relative phases.

ITD overcomes certain limitations of Empirical Mode Decomposition(EMD) including:
overshooting and and undershooting of the interpolating cubic splines generating spurious extrema
distortion and relocation of existing extrema
smearing of time-frequency-energy information
inability or extreme difficulty in producing a correct rotation

https://arxiv.org/pdf/1404.3827v1.pdf good summary of the approach
https://sci-hub.hkvisa.net/10.1098/rspa.2006.1761 original publication
See these articles for more information.

### Available splines

- knot estimation

### Available stopping criteria

-  knot count
-  Fixed number of iterations

### Extrema detection

-  matplotlib findpeaks


## Installation
No package available yet. Work in progress.

## Notes
The repo contains a working ITD implementation in a ipython notebook, and an in progress numba optimized version
