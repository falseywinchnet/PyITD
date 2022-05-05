'''
Intrinsic Time-Scale Decomposition by Frei & Osorio(2006)
Matlab implementation Written by Linshan Jia (jialinshan123@126.com)
Python implementation of baseline,knot finding by Chronum94, detectpeaks adapted from Marcos Duarte
https://arxiv.org/pdf/1404.3827v1.pdf
https://sci-hub.hkvisa.net/10.1098/rspa.2006.1761
NIH/NINDS grants nos. 1R01NS046602-01 and 1R43NS39240-01.
algorithm completion status: 100%
This algorithm is patented US7966156B1 by Frei And Osorio, 2024-10-06 Adjusted expiration
'''
import numpy
import numba

@numba.njit(parallel=True)
def isin(a, b):
    out=numpy.empty(a.shape[0], dtype=numba.boolean)
    b = set(b)
    for i in numba.prange(a.shape[0]):
        if a[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out

@numba.jit(numba.int64[:](numba.float64[:]))
def detect_peaks(x: list[float]):
    """Detect peaks in data based on their amplitude and other features.
    warning: this code is an optimized copy of the "Marcos Duarte, https://github.com/demotu/BMC"
    matlab compliant detect peaks function intended for use with data sets that only want
    rising edge and is optimized for numba. experiment with it at your own peril.
    """
    # find indexes of all peaks
    x = numpy.asarray(x)
    if len(x) < 3:
        return numpy.zeros(3, numpy.int64)
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = numpy.where(numpy.isnan(x))[0]
    indl = numpy.asarray(indnan)

    if indl.size!= 0:
        x[indnan] = numpy.inf
        dx[numpy.where(numpy.isnan(dx))[0]] = numpy.inf

    vil = numpy.zeros(dx.size + 1)
    vil[:-1] = dx[:]# hacky solution because numba does not like hstack tuple arrays
    #numpy.asarray((dx[:], [0.]))# hacky solution because numba does not like hstack
    vix = numpy.zeros(dx.size + 1)
    vix[1:] = dx[:]

    ind = numpy.unique(numpy.where((vil <= 0) & (vix > 0))[0])
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
    return ind

@numba.jit(numba.float64[:,:](numba.float64[:]))
def ITD(data: list[numpy.float64]):
    data = numpy.transpose(numpy.asarray(data,dtype=numpy.float64))
    Lx, Hx, cont = itd_baseline_extract(data)
    H = numpy.asarray(Hx)
    counter = 1
    while cont:
        Lx, Hx, cont = itd_baseline_extract(Lx)
        H = numpy.vstack((H,numpy.asarray(Hx)))
        counter = counter + 1
        if counter > 10:
            break#we've gone on long enough
    H = numpy.vstack((H,numpy.asarray(Lx)))
    return H

@numba.jit(numba.types.Tuple((numba.float64[:],numba.float64[:],numba.int64))(numba.float64[:]))
def itd_baseline_extract(data: list[numpy.float64]):
    #we do tons of get data 2 new empty array 3 a[:] = b[:] to do some simple memory reorganization
    r = numpy.asarray(data)
    x = numpy.zeros(r.shape[0],dtype=numpy.float64)
    t =  x.shape[0] - 1
    x[:] = r[:]
    alpha=0.5

    idx_max_a = numpy.asarray(detect_peaks(x))
    idx_max = numpy.zeros(idx_max_a.shape[0],dtype=numpy.int64)
    idx_max[:] = idx_max_a[:]

    for each in range(idx_max.shape[0]): #test each index
            while (idx_max[each] != t): #while the index isn't exceeding the bounds and we havn't tested
                if (x[idx_max[each]] == x[idx_max[each] + 1]): #if the indexed value is the same as it's rightmost neighbor
                    idx_max[each] = idx_max[each] + 1 #increment the value by 1
                else:
                     break # we have reached the rightmost value

    val_max = numpy.zeros(idx_max.shape[0],dtype=numpy.float64)
    val_max[:] = x[idx_max[:]] #get positive extrema values based on indexes

    idx_min_a= numpy.asarray(detect_peaks(-x))
    idx_min = numpy.zeros(idx_min_a.shape[0], dtype=numpy.int64)
    idx_min[:] = idx_min_a[:]
    for each in range(idx_min.shape[0]): #test each index
            while (idx_min[each] != t): #while the index isn't exceeding the bounds and we havn't tested
                if (x[idx_min[each]] == x[idx_min[each] + 1]): #if the indexed value is the same as it's rightmost neighbor
                    idx_min[each] = idx_min[each] + 1 #increment the value by 1
                else:
                     break # we have reached the rightmost value

    #https://arxiv.org/pdf/1404.3827v1.pdf page 3 - always take right-most sample
    val_min = numpy.zeros(idx_min.shape[0],dtype=numpy.float64)

    val_min[:] = x[idx_min]
    val_min= -val_min

    rotation = numpy.zeros(data.shape[0],dtype=numpy.float64)

    num_extrema = val_max.size + val_min.size
    if(num_extrema<3):
        return x,rotation, 0 #trend decomposition is complete - there are no more knots to consider.
    indicecount = num_extrema + 2
    extrema_indices = numpy.zeros(indicecount, dtype=numpy.int64)
    origin_indices = numpy.sort(numpy.unique(numpy.hstack((idx_max,idx_min_))))

    extrema_indices[1:-1] = origin_indices[:]
    extrema_indices[-1] = len(x) - 1

    baseline_knots = numpy.zeros((num_extrema),dtype=numpy.flooat64)
    baseline_knots[0] = numpy.mean(x[:2])
    baseline_knots[-1] = numpy.mean(x[-2:])
    #also reflections possible, but should be treated with caution

    #j = extrema_indices, k = k, baseline_knots = B, x =  τ
    for k in range(1, len(extrema_indices) - 1):
        baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] + \
        (extrema_indices[k] - extrema_indices[k - 1]) / (extrema_indices[k + 1] - extrema_indices[k - 1]) * \
        (x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]])) + \
                         alpha * x[extrema_indices[k]]

    baseline_new = numpy.zeros(x.shape[0],dtype=numpy.float64)
    #baseline = b^j+1, x = bj(previous baseline)
    for k in range(0, extrema_indices.shape[0] - 1):
        baseline_new[extrema_indices[k]:extrema_indices[k + 1]] = baseline_knots[k]  + \
       (baseline_knots[k + 1] - baseline_knots[k]) / (x[extrema_indices[k + 1]] - x[extrema_indices[k]]) * \
       (x[extrema_indices[k]:extrema_indices[k + 1]] - x[extrema_indices[k]])

    rotation = numpy.subtract(x, baseline_new)

    return baseline_new , rotation, 1


r =numpy.array([1,2,3,4,5]).astype(dtype=float)
x = ITD(r)
