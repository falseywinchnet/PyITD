import numpy
import numba
import numpy as np
import scipy.interpolate as interpolate
from math import factorial







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

# recommendations to improve ITD are also implemented here. In particular:
# mirroring? I think i'm doing it right? extrema
# cubic spline interpolation instead of affine smoothing transformation for baseline interpolation
# akima interpolation also gave interesting results, more like affine, perhaps better
# RMS of PRC calculations
# if i understand the paper correctly, omega = the square root of the mean square of ( current rotation - the RMS of the signal)
# the omega for each should be stored, and then the coefficients with the highest omega selected.
# if i understand right, also, a variable is set, let's say b, value not given in the paper
# and then the sum of the rotation is taken. when the sum of 0.005 * the rotation  < b < the sum of 0.05* the rotation,
# the EITD authors considered it decomposed.  #it's also possible to determine how 'proper" a rotation is by orthonormality.

# Adaptive carrier fringe pattern enhancement for
# wavelet transform profilometry through modifying
# intrinsic time-scale decomposition
# Hanxiao Wang,1 Yinghao Miao,1 Hailu Yang,1 Zhoujing Ye,1 AND Linbing Wang2,3,
# proposed MITD in 2020:
# MITD proposes adding white noise with the same deviation but opposite polarity to two copies of S.
# this can be futher increased to multiple pairs.
# each modified signal copy is decomposed using cubic interpolation(as we have here)
# where each of the PRC's are added together with their opposite twin, same for the residual.
# in the case of an ITD instance generating MORE PRC's than the other, MITD proposes padding with PRC's of 0s to reach
# equal copy counts.
# In theory, this could be applied to EITD.
# the decomposition is quite clean and looks good, and doesn't add a huge amount of overhead.
# The reconstruction error is < 10-16 which is as good as numpy.sum can resolve or better.

# then, MITD proposes WPE measurement and selection of PRC based on WPE, with low WPE = less likely to be noise.
# EITD proposed measuring omega and selecting for high omega.
# EITD-MP proposes matching persuit with binary search.

# then, MITD proposes grey correlation and fuzzy similarity.
# EITD proposes wavelet transforms.

# ITD -> Selection -> Transform/Correlate -> extract

# WPE = weighted-permutation entropy. it is implemented here copied from pyEntropy
# it can be used with weighted_permutation_entropy(x), assuming MITD authors didn't intend a different kind of entropy.
# they measure it between 0 and 1 in most cases, which may mean that the normalization=True must be enabled.
# noise is considered to be WPE above 0.6.
# https://github.com/rsarai/grey-relational-analysis/blob/master/Gait%20and%20Grey%20Methods.ipynb its possible
# that some of te code here could be used to implement the grey fuzzy analysis


# all summed up, here in this `paper` i propose to implement MEITD:
# modified Ensemble Intrinsic Decomposition, where each time a rotation is to be repeatedly decomposed,
# we add noise to it before decomposing and perform the paired decomposition.
# this is only a theoretical possibility atm.


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


@numba.njit(numba.int64[:](numba.float64[:]))
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

    ind = numpy.unique(numpy.where((vil <= 0) & (vix > 0))[0])

    rx = numpy.append(dx, [dx[-1] + 1])
    arr_diff = numpy.diff(rx)
    res_mask = arr_diff == 0
    arr_diff_zero_right = numpy.nonzero(res_mask)[0] + 1
    res_mask[arr_diff_zero_right] = True
    repeating = numpy.nonzero(res_mask)[0]
    rset = set(repeating)
    if len(repeating) != 0:  # if there are repeating elements:
        for each in range(len(ind)):
            if ind[each] in rset:  # is this a repeating index?
                ind[each] = numpy.argmax(dx[ind[each]:] != dx[ind[each]]) - 1  # if so, set it to the rightmost value.
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
    return numpy.unique(ind)


@numba.njit(numba.int64[:](numba.float64[:]))
def detect_peaks(x: list[numpy.float64]):
    # warning: this is not a good, proper, peak-finding method.
    # all this does is determine extrema location and counts for ITD.
    # if you want a matlab findpeaks, use Marcos Duarte's findpeaks.
    # if you want a good general purpose peakfinding method, use a guassian method.

    f = x.copy()
    locmax = numpy.zeros_like(f)

    for i in range(1, len(x) - 1):  # don't consider an end value a peak.
        if (f[i] >= f[i - 1]):
            if (f[i] > f[i + 1]):
                if (f[i] > 0):
                    locmax[i] = 1  # strictly rising peaks
    return numpy.where(locmax == 1)[0]


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

    return baseline_knots[:]



def itd_baseline_extract(data: list[numpy.float64]) -> [numpy.ndarray, numpy.ndarray]:
    proper_rotation = 0
    x = numpy.asarray(data, dtype=numpy.float64)
    rotation = numpy.zeros_like(x)
    baseline_new = numpy.zeros_like(x)

    idx_max = numpy.asarray(detect_peaks(x))
    idx_min = numpy.asarray(detect_peaks(-x))

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


def retrieve_proper_rotation(x: numpy.ndarray):
    x = numpy.asarray(x).astype(dtype=numpy.float64)
    rotation_ = numpy.zeros((len(x)), dtype=numpy.float64)
    baseline_ = numpy.zeros((len(x)), dtype=numpy.float64)
    beta = math.fsum(x)
    idx_max = numpy.asarray(matlab_detect_peaks(x))
    idx_min = numpy.asarray(matlab_detect_peaks(-x))
    num_extrema = idx_min.size + idx_max.size
    baseline_[:] = x[:] #we start with the rotation and we take it from there
    if num_extrema < 5:
        print("I can't decompose this!")
        return x, 0
    else:
        while num_extrema > 5:
            rotation_[:], baseline_[:] = itd_baseline_extract(baseline_[:])
            L = abs(math.fsum(rotation_) * 0.005)
            R = abs(math.fsum(rotation_) * 0.05)
            idx_max = numpy.asarray(detect_peaks(baseline_[:]))
            idx_min = numpy.asarray(detect_peaks(-baseline_))
            num_extrema = idx_min.size + idx_max.size
            beta = 0.01
            if L < beta and beta < R:
                return rotation_[:], 1
        # iteratively and repeatedly decompose this mode until a proper rotation is found-
        # ideally, the first proper rotation!
        return x, 0


def determine_if_first_is_proper_rotation(x: numpy.ndarray):
    x = numpy.asarray(x).astype(dtype=numpy.float64)
    rotation_ = numpy.zeros((len(x)), dtype=numpy.float64)
    baseline_ = numpy.zeros((len(x)), dtype=numpy.float64)
    idx_max = numpy.asarray(matlab_detect_peaks(x))
    idx_min = numpy.asarray(matlab_detect_peaks(-x))
    num_extrema = idx_min.size + idx_max.size
    if num_extrema < 5:
        return x , rotation_ , 0
    else:
        rotation_[:], baseline_[:] = itd_baseline_extract(x[:])
        L = abs(math.fsum(rotation_) * 0.005)
        R = abs(math.fsum(rotation_) * 0.05)
        idx_max = numpy.asarray(matlab_detect_peaks(baseline_[:]))
        idx_min = numpy.asarray(matlab_detect_peaks(-baseline_))
        num_extrema = idx_min.size + idx_max.size
        beta = 0.01
        if L < beta and beta < R:
            return rotation_[:], baseline_[:], 1
        else:
            return rotation_[:], baseline_[:], 0
    # decompose rotations from signal evaluating decompositional potential

def MEITD(data: numpy.ndarray, max_iteration: int = 22) -> numpy.ndarray:
    x = numpy.asarray(data).astype(dtype=numpy.float64)
    highrotations = numpy.zeros((22, len(data)), dtype=numpy.float64)
    lowrotations = numpy.zeros((22, len(data)), dtype=numpy.float64)
    highcounter = 0
    lowcounter = 0
    zero_sum = numpy.zeros((len(data)), dtype=numpy.float64)

    rotations = numpy.zeros((45, len(data)), dtype=numpy.float64)
    rotation_ = numpy.zeros((len(data)), dtype=numpy.float64)
    baseline_ = numpy.zeros((len(data)), dtype=numpy.float64)
    rotation_[:], baseline_[:], proper_rotation = determine_if_first_is_proper_rotation(x)
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
                rotation_[:], proper_rotation = retrieve_proper_rotation(rotation_[:])

            if proper_rotation == 1:
                # so, we either got lucky on our first try, or we retrieved a proper rotation just now.
                if HILO == 1:
                    highrotations[highcounter, :] = rotation_[:]
                    highcounter = highcounter + 1
                    # if HILO is 1, we have decomposed a higher end rotation.
                else:
                    lowrotations[lowcounter, :] = rotation_[:]
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
                rotation_[:], rt, proper_rotation = determine_if_first_is_proper_rotation(baseline_[:])
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
                rotation_[:],cr, proper_rotation = determine_if_first_is_proper_rotation(baseline_[:])
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
                rotation_[:], baseline_[:], proper_rotation = determine_if_first_is_proper_rotation(x)
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
    highrotations = numpy.zeros((40,45, len(data)), dtype=numpy.float64)
    highcounter = 0
    lowrotations = numpy.zeros((40,45, len(data)), dtype=numpy.float64)
    lowcounter = 0
    residual = numpy.zeros((len(data)), dtype=numpy.float64)
    highrotations_, lowrotations_ , residual_ = MEITD(data)
    xww = 0 
    highrotations[0,0:highrotations_.shape[0],:] = highrotations_[:]
    lowrotations[0,0:lowrotations_.shape[0],:] = lowrotations_[:]
    residual[:] = residual_[:]
    counter = 0
    iteration = 0
    not_finished = 1
    while not_finished == 1:
        not_finished = 0
        if (highrotations_.shape[0] > 2):
            for each in range(highrotations_.shape[0]):
                highrotationsx_, lowrotationsx_ , residualx_ = MEITD(highrotations_[each,:])
                #we were able to decompose it further. even more further!
                if (highrotationsx_.shape[0] + lowrotationsx_.shape[0]) > 1:
                    highrotations[iteration,each,:] = 0 #wipe the data
                    counter = counter + 1
                    highrotations[counter,0:highrotationsx_.shape[0],:] = highrotationsx_[:]
                    lowrotations[counter,0:lowrotationsx_.shape[0],:] = lowrotationsx_[:]
                    residual[:] = residual[:]  + residualx_[:]
        if (lowrotations_.shape[0] > 2):
            for each in range(lowrotations_.shape[0]):
                highrotationsx_, lowrotationsx_ , residualx_ = MEITD(lowrotations_[each,:])
                #we were able to decompose it further. even more further!
                if (highrotationsx_.shape[0] + lowrotationsx_.shape[0]) > 1:
                    lowrotations[iteration,each,:] = 0 #wipe the data

                    counter = counter + 1
                    #not_finished == 1
                    highrotations[counter,0:highrotationsx_.shape[0],:] = highrotationsx_[:]
                    lowrotations[counter,0:lowrotationsx_.shape[0],:] = lowrotationsx_[:]
                    residual[:] = residual[:]  + residualx_[:]        
        highrotations_, lowrotations_ , residual_ =    MEITD(residual[:])  
        
        q = highrotations_.shape[0] + lowrotations_.shape[0]
        
        if (highrotations_.shape[0] + lowrotations_.shape[0]) > 2: 
            not_finished = 1
            iteration = counter +1 #store the results in iteration
            counter = iteration + 1
            highrotations[iteration,0:highrotations_.shape[0],:] = highrotations_[:]
            lowrotations[iteration,0:lowrotations_.shape[0],:] = lowrotations_[:]
            residual[:] = residual_[:] #reset the residual trend
        if q == xww:
            not_finished = 0
            print("finished!")
        xww = q
        
        
    
    highrotations = highrotations.reshape((-1, 8000))
    highrotations = highrotations[np.all(highrotations!=0, axis=1)]#take only the rows which are nonzero
    highrotations = numpy.unique(highrotations,axis=0)

    
    lowrotations = lowrotations.reshape((-1, 8000)) 
    lowrotations = lowrotations[np.all(lowrotations!=0, axis=1)]
    lowrotations = numpy.unique(lowrotations,axis=0)
    
    rotations = numpy.vstack((highrotations,lowrotations))
    ent = []
    for i in range(rotations.shape[0]):
        ent.append(weighted_permutation_entropy(rotations[i,:], order=3, normalize=True))
    rotations = rotations[np.argsort(ent), :]
    
    rotations = numpy.vstack((rotations,residual))
    #rotations = rotations[np.all(rotations!=0, axis=1)]#take only the rows which are nonzero
    #rotations = numpy.unique(rotations,axis=0)


    #rotations = numpy.zeros(((highrotations.shape[0]+lowrotations.shape[0]+1,residual_.shape[0])),dtype=numpy.float64)
    #rotations[0:highrotations.shape[0],:]= highrotations[:]
    #rotations[highrotations.shape[0]:-1,:]= lowrotations[:]
    #rotations[-1,:] = residual[:]
    #rotations = numpy.unique(rotations,axis=0)
    return rotations
        
