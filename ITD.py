import numpy
import numba

@numba.njit(numba.boolean[:](numba.int64[:],numba.int64[:]),parallel=True)
def isin(a, b):
    out=numpy.empty(a.shape[0], dtype=numba.boolean)
    b = set(b)
    for i in numba.prange(a.shape[0]):
        if a[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out

@numba.njit(numba.int64[:](numba.float64[:]))
def detect_peaks(x: list[float]):
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

    if indl.size!= 0:
        x[indnan] = numpy.inf
        dx[numpy.where(numpy.isnan(dx))[0]] = numpy.inf

    vil = numpy.zeros(dx.size + 1)
    vil[:-1] = dx[:]# hacky solution because numba does not like hstack tuple arrays
    #numpy.asarray((dx[:], [0.]))# hacky solution because numba does not like hstack
    vix = numpy.zeros(dx.size + 1)
    vix[1:] = dx[:]

    ind = numpy.unique(numpy.where((vil <= 0) & (vix > 0))[0])

            
    rx = numpy.append(dx,[dx[-1] + 1])
    arr_diff = numpy.diff(rx)
    res_mask = arr_diff == 0
    arr_diff_zero_right = numpy.nonzero(res_mask)[0] + 1
    res_mask[arr_diff_zero_right] = True
    repeating = numpy.nonzero(res_mask)[0]
    rset = set(repeating)
    if len(repeating)!= 0: #if there are repeating elements:
        for each in range(len(ind)):
            if ind[each] in rset: #is this a repeating index?
                ind[each] = numpy.argmax(dx[ind[each]:] != dx[ind[each]]) -1 #if so, set it to the rightmost value.
    #this adjustment is intended to implement "rightmost value of flat peaks" efficiently.    
       #https://arxiv.org/pdf/1404.3827v1.pdf page 3 - always take right-most sample
        
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
        
    #eliminate redundant values
    
    
    return numpy.unique(ind)

@numba.jit(numba.float64[:,:](numba.float64[:]))
def ITD(data: list[numpy.float64]):
    Lx, Hx, cont = itd_baseline_extract(numpy.transpose(numpy.asarray(data,dtype=numpy.float64)))
    H = numpy.zeros((22,len(data)),dtype=numpy.float64)
    H[0,:] = Hx[:]
    counter = 1
    while cont:
        Lx, Hx, cont = itd_baseline_extract(Lx)
        H[counter,:] = Hx[:]
        counter = counter + 1
        if counter > 20:
            counter = counter - 1 
            break#we've gone on long enough
    counter = counter + 1
    H[counter,:] = Lx[:]
    return H[0:counter+1,:]

@numba.jit(numba.types.Tuple((numba.float64[:],numba.float64[:],numba.int64))(numba.float64[:]))
def itd_baseline_extract(data: list[numpy.float64]):

    x = numpy.asarray(data,dtype=numpy.float64)
    rotation = numpy.zeros(x.shape[0],dtype=numpy.float64)

    t =  x.shape[0] - 1
    alpha=0.5
    # signal.find_peaks_cwt(x, 1)
    idx_max =numpy.asarray(detect_peaks(x))
    idx_min= numpy.asarray(detect_peaks(-x))
    shared = numpy.intersect1d(idx_max, idx_min)
    if len(shared) != 0:
        print("Danger! how can peaks and valleys be the same?")
        return x,rotation, 0 #trend decomposition is broken

    val_max = x[idx_max] #get peaks based on indexes
    val_min = x[idx_min]
    val_min= -val_min
    
    num_extrema = len(val_max) + len(val_min)
    if(num_extrema<3):
        return x,rotation, 0 #trend decomposition is complete - there are no more knots to consider.
    extremabuffersize = num_extrema + 2
    extrema_indices = numpy.zeros(extremabuffersize, dtype=numpy.int64)
    extrema_indices[1:-1] = numpy.sort(numpy.unique(numpy.hstack((idx_max,idx_min)))) 
    extrema_indices[-1] = len(x) - 1
    
    baseline_knots = numpy.zeros(len(extrema_indices))
    baseline_knots[0] = numpy.mean(x[:2])
    baseline_knots[-1] = numpy.mean(x[-2:])
    #also reflections possible, but should be treated with caution

    #j = extrema_indices, k = k, baseline_knots = B, x =  τ
    for k in range(1, len(extrema_indices) - 1):
        baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] + \
        (extrema_indices[k] - extrema_indices[k - 1]) / (extrema_indices[k + 1] - extrema_indices[k - 1]) * \
        (x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]])) + \
                            alpha * x[extrema_indices[k]]
    
    #q * (n + ((x - y)/(z - y)) * (v - n)) + q * o = 9 -> 3 mult, 3 sub, 1 div, 2 add
    #q * (n + o + ((n - v) * (x - y))/(y - z)) =  8 -> 2 mult, 3 sub, 1 div, 2 add
    #algebraically, these things do the same thing.
    
    #for k in range(1, len(extrema_indices) - 1):
     #       baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] + \
      #      x[extrema_indices[k]] + ((extrema_indices[k] - extrema_indices[k - 1]) * x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]]) / \
       #                              (extrema_indices[k + 1] - extrema_indices[k - 1]))
   
    #using wolfram alpha and remapping indexes to algebra variables, found a slight improvement saving one instruction.
    #However, it does not output similar results at all!
    
    baseline_new = numpy.zeros_like(x)
    #baseline = b^j+1, x = bj(previous baseline)
    for k in range(0, len(extrema_indices) - 1):
        baseline_new[extrema_indices[k]:extrema_indices[k + 1]] = baseline_knots[k]  + \
       (baseline_knots[k + 1] - baseline_knots[k]) / (x[extrema_indices[k + 1]] - x[extrema_indices[k]]) * \
       (x[extrema_indices[k]:extrema_indices[k + 1]] - x[extrema_indices[k]])
    
    rotation[:] = numpy.subtract(x, baseline_new)

    return baseline_new , rotation, 1
