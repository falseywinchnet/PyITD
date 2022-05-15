import numpy
import numba
import numpy as np
import scipy.interpolate as interpolate
from math import factorial

#The following functions are implemented/stolen here:
#numba accelerated cubic interpolation(except the spline construction, splrep, not sure about time cost there
#two different findpeaks functions: not sure which to use
#the matlab style version will always provide rotations which meet frei-osorio "proper rotation" criteria.
#however, these rotations are often not orthonormal and the non-"proper rotation" findpeaks will return
#rotations with extrema translated below and above 0, but which better fit the data.
#I am going to personally go with the second findpeak, but if you are interested in statistically correct answers,
#just switch it out for matlab_detect_peaks

#ensemble ITD "        Hu, Aijun; Yan, Xiaoan; Xiang, Ling  (2015).
# A new wind turbine fault diagnosis method based on ensemble intrinsic time-scale decomposition
# and WPT-fractal dimension. Renewable Energy, 83(), 767–778.
# doi:10.1016/j.renene.2015.04.063 
# is partially implemented here.
#EITD-MP "Wang, Xiaoling; Ling, Bingo Wing-Kuen (2019).
# Underlying Trend Extraction via Joint Ensemble Intrinsic Timescale Decomposition Algorithm and Matching Pursuit Approach.
# Circuits, Systems, and Signal Processing, (), –. doi:10.1007/s00034-019-01069-2
#https://sci-hub.hkvisa.net/10.1016/j.renene.2015.04.063#
#https://sci-hub.hkvisa.net/10.1007/s00034-019-01069-2
#

# recommendations to improve ITD are also implemented here. In particular:
#mirroring? I think i'm doing it right? extrema
#cubic spline interpolation instead of affine smoothing transformation for baseline interpolation
#akima interpolation also gave interesting results, more like affine, perhaps better
#RMS of PRC calculations
#if i understand the paper correctly, omega = the square root of the mean square of ( current rotation - the RMS of the signal)
#the omega for each should be stored, and then the coefficients with the highest omega selected.
#if i understand right, also, a variable is set, let's say b, value not given in the paper
#and then the sum of the rotation is taken. when the sum of 0.005 * the rotation  < b < the sum of 0.05* the rotation,
# the EITD authors considered it decomposed.  #it's also possible to determine how 'proper" a rotation is by orthonormality.

#Adaptive carrier fringe pattern enhancement for
#wavelet transform profilometry through modifying
#intrinsic time-scale decomposition
#Hanxiao Wang,1 Yinghao Miao,1 Hailu Yang,1 Zhoujing Ye,1 AND Linbing Wang2,3,
# proposed MITD in 2020:
#MITD proposes adding white noise with the same deviation but opposite polarity to two copies of S.
#this can be futher increased to multiple pairs.
#each modified signal copy is decomposed using cubic interpolation(as we have here)
#where each of the PRC's are added together with their opposite twin, same for the residual.
#in the case of an ITD instance generating MORE PRC's than the other, MITD proposes padding with PRC's of 0s to reach
#equal copy counts.
#In theory, this could be applied to EITD.
#the decomposition is quite clean and looks good, and doesn't add a huge amount of overhead.
#The reconstruction error is < 10-16 which is as good as numpy.sum can resolve or better.

#then, MITD proposes WPE measurement and selection of PRC based on WPE, with low WPE = less likely to be noise.
#EITD proposed measuring omega and selecting for high omega.
#EITD-MP proposes matching persuit with binary search.

#then, MITD proposes grey correlation and fuzzy similarity.
#EITD proposes wavelet transforms.

#ITD -> Selection -> Transform/Correlate -> extract

#WPE = weighted-permutation entropy. it is implemented here copied from pyEntropy
#it can be used with weighted_permutation_entropy(x), assuming MITD authors didn't intend a different kind of entropy.
#they measure it between 0 and 1 in most cases, which may mean that the normalization=True must be enabled.
#noise is considered to be WPE above 0.6.
#https://github.com/rsarai/grey-relational-analysis/blob/master/Gait%20and%20Grey%20Methods.ipynb its possible
#that some of te code here could be used to implement the grey fuzzy analysis




#all summed up, here in this `paper` i propose to implement MEITD:
#modified Ensemble Intrinsic Decomposition, where each time a rotation is to be repeatedly decomposed,
#we add noise to it before decomposing and perform the paired decomposition.
#secondly, we extract and preserve the 0mega and WPE measurements for each first proper rotation extracted.
#We select the components with 0mega above 0.01 and WPE below 0.06 and then display or re-decompose them.

#this is only a theoretical possibility atm.





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


def itd_baseline_extract(data: list[numpy.float64]) -> [numpy.ndarray, numpy.ndarray, numpy.int64]:
    proper_rotation = 0
    x = numpy.asarray(data, dtype=numpy.float64)
    rotation = numpy.zeros_like(x)
    baseline_new = numpy.zeros_like(x)

    idx_max = numpy.asarray(detect_peaks(x))
    idx_min = numpy.asarray(detect_peaks(-x))

    common_values = set(idx_max) & set(idx_min)

    a = idx_max.tolist()
    b = idx_min.tolist()

    for value in common_values:
        a.remove(value)
        b.remove(value)
    # why are there common values!? ERROR!

    idx_max = numpy.array(a)
    idx_min = numpy.array(b)

    num_extrema = idx_min.size + idx_max.size

    extremabuffersize = num_extrema + 2
    extrema_indices = numpy.zeros(extremabuffersize, dtype=numpy.int64)
    extrema_indices[1:-1] = numpy.sort(numpy.unique(numpy.hstack((idx_max, idx_min))))
    extrema_indices[-1] = len(x) - 1

    baseline_knots = numpy.zeros(len(extrema_indices))

    padded = numpy.pad(x, 1, mode='reflect', reflect_type="even")
    baseline_knots[0] = numpy.mean(padded[:2])
    baseline_knots[-1] = numpy.mean(padded[-2:])

    baseline_knots[:] = baseline_knot_estimation(baseline_knots[:], x[:], extrema_indices[:])

    z = numpy.zeros_like(x)
    z[:] = range(z.shape[0])
    coeff = custom_splrep(extrema_indices, baseline_knots)
    baseline_new[:] = numba_splev(z, coeff)

    rotation[:] = numpy.subtract(x, baseline_new)

    MSE = numpy.square(numpy.subtract(rotation.dot(rotation.T), numpy.eye(rotation.shape[0]))).mean()
    RMSE = numpy.sqrt(MSE)
    print(RMSE)

    if (RMSE) < 6: #just an estimate, I dont know what to set for this
        proper_rotation = 1

    return rotation[:], baseline_new[:], proper_rotation


# @numba.jit(numba.float64[:,:](numba.float64[:]))
def eitd(data: numpy.ndarray, max_iteration: int = 22) -> numpy.ndarray:
    x = numpy.asarray(data).astype(dtype=numpy.float64)
    MSEB = numpy.square(numpy.subtract(x.dot(x.T), numpy.eye(x.shape[0]))).mean()
    RMSEB = numpy.sqrt(MSEB)
    rotations = numpy.zeros((220, len(data)), dtype=numpy.float64)
    baselines = numpy.zeros((220, len(data)), dtype=numpy.float64)
    rotation_ = numpy.zeros((len(data)), dtype=numpy.float64)
    baseline_ = numpy.zeros((len(data)), dtype=numpy.float64)
    counter = 0
    proper_rotation = 0
    rotation_[:], baseline_[:], proper_rotation =  itd_baseline_extract(x)

    idx_max = numpy.asarray(detect_peaks(baseline_))
    idx_min = numpy.asarray(detect_peaks(-baseline_))
    num_extrema = idx_min.size + idx_max.size
    if num_extrema > 4:
        while 1:
            if counter > 20:
                counter = counter + 1
                print("exceeded iterations provided!")
                rotations[counter,:] = x
                return rotations[0:counter+1,:]
            if proper_rotation == 0:
                # So, the first rotation wasn't proper, but it's decomposable. Let's decompose it further.
                rotation_[:], baseline_[:], proper_rotation = itd_baseline_extract(rotation_[:])
                while proper_rotation == 0:
                    idx_max = numpy.asarray(detect_peaks(baseline_))
                    idx_min = numpy.asarray(detect_peaks(-baseline_))
                    num_extrema = idx_min.size + idx_max.size
                    if num_extrema < 4:
                        # could not decompose a proper rotation!!
                        print("Couldn't produce a proper rotation!")
                        counter = counter + 1
                        rotations[counter, :] = x[:]  # store the residual
                        return rotations[0:counter, :]
                    else:
                        rotation_[:], baseline_[:], proper_rotation = itd_baseline_extract(baseline_[:])

                rotations[counter, :] = rotation_[:]  # store the first good rotation
                x = x - rotations[counter, :]  # remove it from the data set
                counter = counter + 1  # increment the counter
                idx_max = numpy.asarray(detect_peaks(baseline_))
                idx_min = numpy.asarray(detect_peaks(-baseline_))
                num_extrema = idx_min.size + idx_max.size
                if num_extrema > 4:
                    rotation_[:], baseline_[:], proper_rotation = itd_baseline_extract(x)
                    continue  # go back to the start, we can continue work
                else:
                    # so, we're out of residual to decompose. No more peaks are possible.
                    # what do we do now?
                    print("Couldn't produce proper rotation from residual!")
                    rotations[counter, :] = x[:]  # get the residual baseline
                    return rotations[0:counter, :]
            else:  # the first/current rotation was proper, so let's extract it.
                rotations[counter, :] = rotation_[:]  # This was  a good initial rotation, just store it
                x = x - rotations[counter, :]
                counter = counter + 1  # increment the counter
                rotation_[:], baseline_[:], proper_rotation = itd_baseline_extract(x)
                continue
    elif proper_rotation == 0:
        print("Couldn't produce any rotation!")
        return x
    else:
        # could only produce one rotation component before peak count too few
        rotations[counter, :] = rotation_[:]
        rotations[counter + 1, :] = baseline_[:]
        counter = counter + 1
        return rotations[0:counter, :]
