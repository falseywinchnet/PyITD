import numpy
import numba
import numpy as np
import scipy.interpolate as interpolate
from math import factorial

"""
MEITD is a non-parametric trend extraction reduction technique based on 
Intrinsic Time-Scale Decomposition, the use of this method for non-research purposes
may be restricted by law due to patent rights on the original ITD algorithm. 
MEITD, Maximal Extraction ensemble Intrinsic-Time-Scale Decomposition
iteratively selects and extracts proper rotations meeting Weighed Permutation Entropy
and seeks to extract a maximum number of components, to leave noise in a residual trend.
The implementation itself is released under the MIT and FSF licenses.

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
"""


#I present Maximal Extraction Intrinsic Time-Scale Decomposition.
#This model is intended to perform maximum extractions, and then sort
#the output by entropy. Noise is generally the very last item-
#the trends from 0:-1 are increasing entropy and frequency.



# The following functions are implemented/stolen here:
# numba accelerated cubic interpolation(except the spline construction, splrep, not sure about time cost there
# two different findpeaks functions: not sure which to use
# the matlab style version will always provide rotations which meet frei-osorio "proper rotation" criteria.
# however, these rotations are often not orthonormal and the non-"proper rotation" findpeaks will return
# rotations with extrema translated below and above 0, but which better fit the data.
# I am going to personally go with the second findpeak, but if you are interested in statistically correct answers,
# just switch it out for matlab_detect_peaks

# ensemble ITD "        Hu, Aijun; Yan, Xiaoan; Xiang, Ling  (2015).
# A new wind turbine fault diagnosis method based on ensemble intrinsic time-scale decomposition
# and WPT-fractal dimension. Renewable Energy, 83(), 767–778.
# doi:10.1016/j.renene.2015.04.063 
# is partially implemented here.
# EITD-MP "Wang, Xiaoling; Ling, Bingo Wing-Kuen (2019).
# Underlying Trend Extraction via Joint Ensemble Intrinsic Timescale Decomposition Algorithm and Matching Pursuit Approach.
# Circuits, Systems, and Signal Processing, (), –. doi:10.1007/s00034-019-01069-2
# https://sci-hub.hkvisa.net/10.1016/j.renene.2015.04.063#
# https://sci-hub.hkvisa.net/10.1007/s00034-019-01069-2
#


def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


def util_rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def weighted_permutation_entropy(time_series, order=3, normalize=False):
    """Calculate the Weighted Permutation Entropy.
    Weighted permutation entropy is based on the regular permutation entropy,
    but puts additional weight on those windows that show a high variability
    in the initial time series.
    Parameters
    ----------
    time_series : list or np.array
        Time series
    order : int
        Order of permutation entropy
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Weighted Permutation Entropy
    References
    ----------
    .. [1] Bilal Fadlallah et al. Weighted-permutation entropy: A complexity
    measure for time series incorporating amplitude information
    https://link.aps.org/accepted/10.1103/PhysRevE.87.022911
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations

    embedded = _embed(x, order=order)
    sorted_idx = embedded.argsort(kind='quicksort')
    weights = np.var(util_rolling_window(x, order), 1)
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    mapping = {}
    for i in np.unique(hashval):
        mapping[i] = np.where(hashval == i)[0]
    weighted_counts = dict.fromkeys(mapping)
    for k, v in mapping.items():
        weighted_count = 0
        for i in v:
            weighted_count += weights[i]
        weighted_counts[k] = weighted_count
    # Associate unique integer to each permutations
    # Return the counts
    # Use np.true_divide for Python 2 compatibility
    weighted_counts_array = np.array(list(weighted_counts.values()))
    p = np.true_divide(weighted_counts_array, weighted_counts_array.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def custom_splrep(x, y, k=3):
    """
    Custom wrap of scipy's splrep for calculating spline coefficients,
    which also check if the data is equispaced.

    """

    # Check if x is equispaced
    x_diff = np.diff(x)
    equi_spaced = all(np.round(x_diff, 5) == np.round(x_diff[0], 5))
    dx = x_diff[0]

    # Calculate knots & coefficients (cubic spline by default)
    t, c, k = interpolate.splrep(x, y, k=k)

    return (t, c, k, equi_spaced, dx)


@numba.njit(cache=True)
def numba_splev(x, coeff):
    """
    Custom implementation of scipy's splev for spline interpolation,
    with additional section for faster search of knot interval, if knots are equispaced.
    Spline is extrapolated from the end spans for points not in the support.

    """
    t, c, k, equi_spaced, dx = coeff

    t0 = t[0]

    n = t.size
    m = x.size

    k1 = k + 1
    k2 = k1 + 1
    nk1 = n - k1

    l = k1
    l1 = l + 1

    y = np.zeros(m)

    h = np.zeros(20)
    hh = np.zeros(19)

    for i in range(m):

        # fetch a new x-value arg
        arg = x[i]

        # search for knot interval t[l] <= arg <= t[l+1]
        if (equi_spaced):
            l = int((arg - t0) / dx) + k
            l = min(max(l, k1), nk1)
        else:
            while not ((arg >= t[l - 1]) or (l1 == k2)):
                l1 = l
                l = l - 1
            while not ((arg < t[l1 - 1]) or (l == nk1)):
                l = l1
                l1 = l + 1

        # evaluate the non-zero b-splines at arg.
        h[:] = 0.0
        hh[:] = 0.0

        h[0] = 1.0

        for j in range(k):

            for ll in range(j + 1):
                hh[ll] = h[ll]
            h[0] = 0.0

            for ll in range(j + 1):
                li = l + ll
                lj = li - j - 1
                if (t[li] != t[lj]):
                    f = hh[ll] / (t[li] - t[lj])
                    h[ll] += f * (t[li] - arg)
                    h[ll + 1] = f * (arg - t[lj])
                else:
                    h[ll + 1] = 0.0
                    break

        sp = 0.0
        ll = l - 1 - k1

        for j in range(k1):
            ll += 1
            sp += c[ll] * h[j]
        y[i] = sp

    return y


@numba.njit(numba.boolean[:](numba.int64[:], numba.int64[:]), parallel=True)
def isin(a, b):
    out = numpy.empty(a.shape[0], dtype=numba.boolean)
    b = set(b)
    for i in numba.prange(a.shape[0]):
        if a[i] in b:
            out[i] = True
        else:
            out[i] = False
    return out


#@numba.njit(numba.int64[:](numba.float64[:]))
def matlab_detect_peaks(x: list[float]):
    """Detect peaks in data based on their amplitude and other features.
    warning: this code is an optimized copy of the "Marcos Duarte, https://github.com/demotu/BMC"
    matlab compliant detect peaks function intended for use with data sets that only want
    rising edge and is optimized for numba. experiment with it at your own peril.
    """
    # find indexes of all peaks
    x = numpy.asarray(x)
    if len(x) < 3:
        return numpy.empty(1, numpy.int64)
    dx = x[1:] - x[:-1]
    dx = -dx
    # handle NaN's
    indnan = numpy.where(numpy.isnan(x))[0]
    indl = numpy.asarray(indnan)

    if indl.size != 0:
        x[indnan] = numpy.inf
        dx[numpy.where(numpy.isnan(dx))[0]] = numpy.inf

    vil = numpy.zeros(dx.size + 1)
    vil[:-1] = dx[:]  # hacky solution because numba does not like hstack tuple arrays
    # numpy.asarray((dx[:], [0.]))# hacky solution because numba does not like hstack
    vix = numpy.zeros(dx.size + 1)
    vix[1:] = dx[:]

    ind = numpy.unique(numpy.where((vil > 0) & (vix <= 0))[0])
    #ind = ind - 1 #gotta shift it back to the left

    # this adjustment is intended to implement "rightmost value of flat peaks" efficiently.
    # https://arxiv.org/pdf/1404.3827v1.pdf page 3 - always take right-most sample

    # handle NaN's
    # NaN's and values close to NaN's cannot be peaks
    if ind.size and indl.size:
        outliers = numpy.unique(numpy.concatenate((indnan, indnan - 1, indnan + 1)))
        booloutliers = isin(ind, outliers)
        booloutliers = numpy.invert(booloutliers)
        ind = ind[booloutliers]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    # eliminate redundant values
    return ind

@numba.jit(cache=True)
def baseline_knot_estimation(baseline_knots: list[numpy.float64], x: list[numpy.float64],
                             extrema_indices: list[numpy.float64]):
    alpha = 0.5
    for k in range(1, len(extrema_indices) - 1):
        baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] \
                                     + ((extrema_indices[k] - extrema_indices[k - 1]) \
                                        / (extrema_indices[k + 1] - extrema_indices[k - 1])) \
                                     * (x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]])) \
                            + (alpha * x[extrema_indices[k]])

    return baseline_knots



def itd_baseline_extract(data: list[numpy.float64]) -> [numpy.ndarray, numpy.ndarray]:
    proper_rotation = 0
    x = numpy.asarray(data, dtype=numpy.float64)
    rotation = numpy.zeros_like(x)
    baseline_new = numpy.zeros_like(x)

    idx_max = numpy.asarray(matlab_detect_peaks(x))
    idx_min = numpy.asarray(matlab_detect_peaks(-x))


    num_extrema = idx_min.size + idx_max.size
    #if num_extrema < 5:
     #   print("decomposition failed in ITD_baseline_extract")
     #   return x, baseline_new, 0
    extremabuffersize = num_extrema + 2
    extrema_indices = numpy.zeros(extremabuffersize, dtype=numpy.int64)
    extrema_indices[1:-1] = numpy.sort(numpy.unique(numpy.hstack((idx_max, idx_min))))
    extrema_indices[-1] = len(x) - 1

    baseline_knots = numpy.zeros(len(extrema_indices))

    padded = numpy.pad(x, 1, mode='reflect', reflect_type="odd")
    baseline_knots[0] = numpy.mean(padded[:2])
    baseline_knots[-1] = numpy.mean(padded[-2:])

    baseline_knots[:] = baseline_knot_estimation(baseline_knots[:], x[:], extrema_indices[:])

    z = numpy.zeros_like(x)
    z[:] = range(z.shape[0])
    coeff = custom_splrep(extrema_indices, baseline_knots)
    baseline_new[:] = numba_splev(z, coeff)

    rotation[:] = numpy.subtract(x, baseline_new)


    return rotation[:], baseline_new[:]


import math


def retrieve_proper_rotation(x: numpy.ndarray,WPEMAX):
    x = numpy.asarray(x).astype(dtype=numpy.float64)
    WPE = weighted_permutation_entropy(x, order=3, normalize=True)
    WPESUM = numpy.mean(WPE)
    rotation_ = numpy.zeros((len(x)), dtype=numpy.float64)
    baseline_ = numpy.zeros((len(x)), dtype=numpy.float64)
    beta = math.fsum(x)
    idx_max = numpy.asarray(matlab_detect_peaks(x))
    idx_min = numpy.asarray(matlab_detect_peaks(-x))
    num_extrema = idx_min.size + idx_max.size
    baseline_[:] = x.copy() #we start with the rotation and we take it from there
    if num_extrema < 5:
        print("I can't retrieve a proper rotation")
        return x, 0
    else:
        while num_extrema > 5:
            rotation_[:], baseline_[:] = itd_baseline_extract(baseline_[:])
            idx_max = numpy.asarray(matlab_detect_peaks(baseline_[:]))
            idx_min = numpy.asarray(matlab_detect_peaks(-baseline_))
            num_extrema = idx_min.size + idx_max.size
            if WPESUM < WPEMAX and not WPESUM < 0.2 : # criteria
                return rotation_[:], 1
        # iteratively and repeatedly decompose this mode until a proper rotation is found-
        # ideally, the first proper rotation!
        return x, 0


def determine_if_first_is_proper_rotation(x: numpy.ndarray,WPEMAX):
    x = numpy.asarray(x).astype(dtype=numpy.float64)
    WPE = weighted_permutation_entropy(x, order=3, normalize=True)
    WPESUM = numpy.mean(WPE)
    rotation_ = numpy.zeros((len(x)), dtype=numpy.float64)
    baseline_ = numpy.zeros((len(x)), dtype=numpy.float64)
    idx_max = numpy.asarray(matlab_detect_peaks(x))
    idx_min = numpy.asarray(matlab_detect_peaks(-x))
    num_extrema = idx_min.size + idx_max.size
    if num_extrema < 5:
        print("I can't retrieve any rotation")

        return x , rotation_ , 0
    else:
        rotation_[:], baseline_[:] = itd_baseline_extract(x[:])
        idx_max = numpy.asarray(matlab_detect_peaks(baseline_[:]))
        idx_min = numpy.asarray(matlab_detect_peaks(-baseline_))
        num_extrema = idx_min.size + idx_max.size
        if WPESUM < WPEMAX and not WPESUM < 0.2: # criteria
            return rotation_[:], baseline_[:], 1
        else:
            return rotation_[:], baseline_[:], 0
    # decompose rotations from signal evaluating decompositional potential

def MEITD(data: numpy.ndarray, max_iteration: int = 40,WPEMAX: float = 0.6) -> numpy.ndarray:
    x = numpy.asarray(data).astype(dtype=numpy.float64)
    highrotations = numpy.zeros((44, len(data)), dtype=numpy.float64)
    lowrotations = numpy.zeros((44, len(data)), dtype=numpy.float64)
    highcounter = 0
    lowcounter = 0
    zero_sum = numpy.zeros((len(data)), dtype=numpy.float64)

    rotations = numpy.zeros((85, len(data)), dtype=numpy.float64)
    rotation_ = numpy.zeros((len(data)), dtype=numpy.float64)
    baseline_ = numpy.zeros((len(data)), dtype=numpy.float64)
    rotation_[:], baseline_[:], proper_rotation = determine_if_first_is_proper_rotation(x,WPEMAX)
    xchanged = 0
    HILO = 1
    soft_reset = 1
    idx_max = numpy.asarray(matlab_detect_peaks(x))
    idx_min = numpy.asarray(matlab_detect_peaks(-x))
    num_extrema = idx_min.size + idx_max.size
    if num_extrema < 4:
        return zero_sum , zero_sum , x
    while num_extrema > 5:
            # TODO : periodically evaluate criteria for inclusion and rebuild rotational base
            # from data with highest omega and WPE below 0.6-
            # removed rotations get integrated into a residual array that X(trend) will be
            # added to at the end of the routine aka thin the herd


            counter = highcounter + lowcounter

            if counter > 20:
                #print("I decomposed ",highcounter ," high IMF's and ", lowcounter , " Low.")
                #print("exceeded iterations provided!")
               # rotations[0:highcounter, :] = highrotations[0:highcounter, :]
               # rotations[highcounter:highcounter + lowcounter, :] = lowrotations[0:lowcounter, :]
                #counter = counter + 1

                #rotations[counter, :] = x
                #return rotations[0:counter + 1, :]
                return highrotations[0:highcounter, :],  lowrotations[0:lowcounter, :], x[:]

            if proper_rotation == 0:
                # So, the first rotation wasn't proper, but it's decomposable. Let's decompose it further.
                rotation_[:], proper_rotation = retrieve_proper_rotation(rotation_[:],WPEMAX)

            if proper_rotation == 1:
                # so, we either got lucky on our first try, or we retrieved a proper rotation just now.
                if HILO == 1:
                    highrotations[highcounter, :] = rotation_.copy()
                    highcounter = highcounter + 1
                    # if HILO is 1, we have decomposed a higher end rotation.
                else:
                    lowrotations[lowcounter, :] = rotation_.copy()
                    lowcounter = lowcounter + 1

                    # if hilo is 0, we have decomposed a lower end rotation.
                    # which, doesn't really matter, maybe. softreset would tell us.

                soft_reset = 0
                x = x - rotation_[:]  # regardlesss, remove it from the data set
                xchanged = 1

            if xchanged == 1 and HILO == 1:
                idx_max = numpy.asarray(matlab_detect_peaks(x))
                idx_min = numpy.asarray(matlab_detect_peaks(-x))
                num_extrema = idx_min.size + idx_max.size
                if num_extrema < 5:
                    continue #break here if we can't decompose
                lol, baseline_[:]  = itd_baseline_extract(x)
                rotation_[:], rt, proper_rotation = determine_if_first_is_proper_rotation(baseline_[:],WPEMAX)
                xchanged = 0  # reset the variable after rebasing
                HILO = 0
                #zt, rt, lol = we want to TOSS these values.
                #that is to say, we want to save the baselines of X-
                #but "retrieve proper rotation" will perform a rotation -> rinse and repeate baseline decomposit
                #and we only want the rotation for that
                continue #go back to the top and attempt to decomopose the rotation here



            elif HILO == 1:
                # we didn't successfully decompose a component, so let's not waste effort.
                #the first time this runs, any time this runs, it will skip the rebase
                rotation_[:],cr, proper_rotation = determine_if_first_is_proper_rotation(baseline_[:],WPEMAX)
                HILO = 0
                continue #go back to the top and reset

            if xchanged == 1 and HILO == 0:
                # we successfully decomposed a lower frequency component.
                # let 's go back again and attempt a high frequency decomposition.
                idx_max = numpy.asarray(matlab_detect_peaks(x))
                idx_min = numpy.asarray(matlab_detect_peaks(-x))
                num_extrema = idx_min.size + idx_max.size
                if num_extrema < 5:
                    continue  # break here if we can't decompose
                rotation_[:], baseline_[:], proper_rotation = determine_if_first_is_proper_rotation(x,WPEMAX)
                xchanged = 0  # reset the variable after rebasing
                HILO = 1
                continue

            if xchanged == 0 and HILO == 0:
                # so, we didn't succeed at decomposing a lower frequency component in this iteration-
                # and we didn't succeed at decomposing a higher frequency component, either.
                # after all, if we had, safety would be zero.
                # at this point, let's try digging.
                if soft_reset == 0:
                    rotation_[:], baseline_[:]  = itd_baseline_extract(x)
                    soft_reset = 1
                idx_max = numpy.asarray(matlab_detect_peaks(baseline_))
                idx_min = numpy.asarray(matlab_detect_peaks(-baseline_))
                num_extrema = idx_min.size + idx_max.size
                if num_extrema < 5:
                    continue #break here if we can't go any further
                for each in range(soft_reset):
                    rotation_[:], baseline_[:]  = itd_baseline_extract(baseline_[:])
                    idx_max = numpy.asarray(matlab_detect_peaks(baseline_))
                    idx_min = numpy.asarray(matlab_detect_peaks(-baseline_))
                    num_extrema = idx_min.size + idx_max.size
                    if num_extrema < 5:
                        break #break here if we can't go any further
                soft_reset = soft_reset + 1
                continue
                # each time we come here, if we didn't succeed at decomposing anything else, we increment further.

                # let's go down another tier.

            # so, what this logic above does, is it iteratively checks to see if it can decompose a
            # proper rotation from X. if it can't, it attempts to decompose the rotation itself.
            # if it can't do that, or, even if it does, it will proceed to subtract the
            # rotation(if successful), rebase, then decompose the baseline.
            # then it will alternate back and forth.
            # if it doesn't get anywhere, it will skinny dip into baseline as far as is needed.

    #counter = highcounter + lowcounter
    #rotations[0:highcounter, :] = highrotations[0:highcounter, :]
    #rotations[highcounter:highcounter + lowcounter, :] = lowrotations[0:lowcounter, :]
    #counter = counter + 1
    #rotations[counter, :] = x[:]
    # so, we're out of things we can decompose, or found we could.

    return highrotations[0:highcounter, :],  lowrotations[0:lowcounter, :], x[:]

def XITD(data: numpy.ndarray):
    data = data.astype(dtype=numpy.float64)
    m_ = data.mean(axis=0)
    sd_ = data.std(axis=0,ddof=0)
    WPEMAX = numpy.log(abs(20*numpy.log10(abs(numpy.where(sd_ == 0, 0, m_/sd_)))))
    #accurately estimate the maximum good rotations
    highrotations, lowrotations , residual = MEITD(data,WPEMAX)
    rotations = numpy.vstack((highrotations,lowrotations))
    rotations = numpy.vstack((rotations,residual))
    ent = []
    for i in range(rotations.shape[0]):
        ent.append(weighted_permutation_entropy(rotations[i,:], order=3, normalize=True))
    rotations = rotations[np.argsort(ent), :]
    return rotations
        
