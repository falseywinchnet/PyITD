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
    vix = numpy.zeros(dx.size + 1)
    vix[1:] = dx[:]

    ind = numpy.unique(numpy.where((vil > 0) & (vix <= 0))[0])

        
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

@numba.jit(numba.types.Tuple((numba.float64[:],numba.float64[:],numba.int64))(numba.float64[:]))
def itd_baseline_extract(data: list[numpy.float64])-> Tuple[numpy.ndarray, numpy.ndarray]:

        x = numpy.asarray(data,dtype=numpy.float64)
        rotation = numpy.zeros_like(x)

        alpha=0.5
        # signal.find_peaks_cwt(x, 1)
        idx_max =numpy.asarray(detect_peaks(x))
        idx_min= numpy.asarray(detect_peaks(-x))
        val_max = x[idx_max] #get peaks based on indexes
        val_min = x[idx_min]
        val_min= -val_min
    
        num_extrema = len(val_max) + len(val_min)

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
    
        baseline_new = numpy.zeros_like(x)

        for k in range(0, len(extrema_indices) - 1):
            baseline_new[extrema_indices[k]:extrema_indices[k + 1]] = baseline_knots[k]  + \
            (baseline_knots[k + 1] - baseline_knots[k]) / (x[extrema_indices[k + 1]] - x[extrema_indices[k]]) * \
            (x[extrema_indices[k]:extrema_indices[k + 1]] - x[extrema_indices[k]])
    
        rotation[:] = numpy.subtract(x, baseline_new)

        return rotation[:] , baseline_new[:]

@numba.jit(numba.float64[:,:](numba.float64[:]))
def itd(self, data: numpy.ndarray, max_iteration: int = 22) -> numpy.ndarray:
        rotations = numpy.zeros((max_iteration+1,len(data)),dtype=numpy.float64)
        baselines = numpy.zeros((max_iteration+1,len(data)),dtype=numpy.float64)
        rotation_ = numpy.zeros((len(data)),dtype=numpy.float64)
        baseline_ = numpy.zeros((len(data)),dtype=numpy.float64)
        r = numpy.zeros((len(data)),dtype=numpy.float64)
        rotation_[:], baseline_[:] = itd_baseline_extract(numpy.transpose(numpy.asarray(data,dtype=numpy.float64))) 
        counter = 0
        while not finished:         
            idx_max = numpy.asarray(detect_peaks(baseline_))
            idx_min = numpy.asarray(detect_peaks(-baseline_))
            num_extrema = len(idx_min) + len(idx_max)
            print(num_extrema)
            if num_extrema < 2:
                #is the new baseline decomposable?
                print("No more decompositions possible")
                #implied: last decomposition was invalid!
                #not always the case, but efforts to decompose the trend which are meaningful
                #require a little adjustment to get the baseline monotonic trend to show properly.
                r[:] = baselines[counter-1,:]
                rotations[counter,:] = r[:]
                counter = counter + 1      
                return  rotations[0:counter,:]

            elif counter > max_iteration:
                print("Out of time!")
                r[:] = numpy.add(rotation_[:],baseline_[:])
                rotations[counter,:] = r[:]
                counter = counter + 1 
                return rotations[0:counter,:]

            else: #results are sane, so perform an extraction.
                rotations[counter,:] = rotation_[:]
                baselines[counter,:] =  baseline_[:]
                rotation_[:],  baseline_[:] = itd_baseline_extract(baseline_[:])
                counter = counter + 1   
