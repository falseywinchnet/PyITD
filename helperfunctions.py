
def shewchuk(a, axis=0):
    '''shewchuck summation of a numpy array.
    #numpy summation is floating point error prone
    '''
    s = numpy.zeros(a.shape[1])
    for i in range(a.shape[1]):
        s[i] = math.fsum(a[:, i])
    return s

def fingerprint(data: numpy.ndarray):
    coeffs = dwtn(data, wavelet='haar')
    coeff = numpy.asarray(list(coeffs.values())).flatten()
    d = fftpack.dct(coeff, axis=0)
    sigma = numpy.sum(d)
    return sigma / 0.6616518484657332

def getsortedindex(data: numpy.ndarray):
    sort = numpy.argsort(data)  # get the sort order
    mean = numpy.mean(data[sort])  # get the statistical mean
    idx = np.searchsorted(data[sort], mean, side="left")#gets the mean.. of the sorted array
    a = data[sort]
    scaled = numpy.interp(a, (a.min(), a.max()), (-6, +6))
    x = np.linspace(0, 1, data.size)
    y = logit(x)
    y[y == -numpy.inf] = -6
    y[y == +numpy.inf] = 6
    z = numpy.corrcoef(scaled, y)
    completeness = z[0,1]
    return sort[idx],completeness 

#as a special statistical product, using additive noise to achieve a spectrum of results
#and then attempting to select the median outcome as the best results works well in practice
#as long as the trend is consistently for the distribution of the outputs to match the logit
#distribution closely. Included are two functions- function fingerprint derives a perceptual fingerprint
#and function getsortedindex returns the median index and a completeness measure.
#as long as completeness measure is above 0.95, the result can be believed as correct.


@numba.jit(cache=True)
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
        pe /= np.log2(math.factorial(order))
    return pe