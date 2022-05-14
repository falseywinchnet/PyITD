import numpy
import numba

#warning: this code does not work, it isn't even valid python in a sense.
#this was an attempt to create a 2d version of ITD, but I lack the ability to take the equation up a dimension.

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

#@numba.njit(numba.int64[:](numba.float64[:]))
def detect_peaks_2d(x: numpy.ndarray):
    # find indexes of all peaks
    x = numpy.asarray(x)
    
    if x.shape[0] < 3 or x.shape[1] < 3:
        return numpy.empty(((1,1)), dtype=numpy.int64)
    
    
    indnan = numpy.where(numpy.isnan(x))[0]
    indl = numpy.asarray(indnan)
    
    dx = numpy.zeros(x.shape,dtype=numpy.int64)

    if indl.size!= 0:
        x[indnan] = numpy.inf
        dx[numpy.where(numpy.isnan(dx))[0]] = numpy.inf

   
    
  
    dx = numpy.zeros(x.shape,dtype=numpy.int64)
    for a in range(1,x.shape[0]-1):
        for b in range(1,x.shape[1]-1):
            if x[a,b] > 0 and\
             x[a,b-1] < x[a,b] and\
             x[a-1,b] < x[a,b] and\
             x[a-1,b-1] < x[a,b] and\
             x[a,b+1] < x[a,b] and\
             x[a+1,b+1] < x[a,b] and\
             x[a-1,b+1] < x[a,b] and\
             x[a+1,b-1] < x[a,b] and\
             x[a+1,b] < x[a,b]:                       
                dx[a,b] = 1
   
    ind = numpy.argwhere(dx == 1)
    # handle NaN's
    # NaN's and values close to NaN's cannot be peaks
    if ind.size and indl.size:
        outliers = numpy.unique(numpy.concatenate((indnan, indnan - 1, indnan + 1)))
        booloutliers = isin(ind, outliers)
        booloutliers = numpy.invert(booloutliers)
        ind = ind[booloutliers]

    return ind

#@numba.jit(numba.types.Tuple((numba.float64[:],numba.float64[:],numba.int64))(numba.float64[:]))
def itd_baseline_extract_2d(data: numpy.ndarray):

        x = numpy.asarray(data,dtype=numpy.float64)
        rotation = numpy.zeros_like(x)

        alpha=0.5
        f = numpy.asarray(x)
    
        locmax = numpy.zeros_like(f)
        for i in range(1,f.shape[0]-1):#iterate over columns 
            for j in range(1,f.shape[1]-1): #iterate over rows
                if f[i,j] > 0 and (f[i-1,j-1] < f[i,j]) and (f[i,j]>f[i+1,j+1]) and \
                 (f[i-1,j] < f[i,j]) and (f[i,j]>f[i+1,j]) and \
                 (f[i,j-1] < f[i,j]) and (f[i,j]>f[i,j+1]) and \
                    (f[i+1,j-1] < f[i,j]) and (f[i,j] >f[i+1,j-1]) and \
                    (f[i-1,j+1] < f[i,j]): #consider all 9 neighbors
                    locmax[i,j]=f[i,j] 
                
        f = numpy.asarray(-x)
    
        locmin = numpy.zeros_like(f)
        for i in range(1,f.shape[0]-1):#iterate over columns 
            for j in range(1,f.shape[1]-1): #iterate over rows
                if f[i,j] > 0 and (f[i-1,j-1] < f[i,j]) and (f[i,j]>f[i+1,j+1]) and \
                 (f[i-1,j] < f[i,j]) and (f[i,j]>f[i+1,j]) and \
                 (f[i,j-1] < f[i,j]) and (f[i,j]>f[i,j+1]) and \
                    (f[i+1,j-1] < f[i,j]) and (f[i,j] >f[i+1,j-1]) and \
                    (f[i-1,j+1] < f[i,j]): #consider all 9 neighbors
                    locmin[i,j]=f[i,j] 
                
                
        extremaval = locmax + locmin #perform an in-place combination, zeros will add to zero
        #intrinsically, these are already sorted.
        
        extremaval1 = extremaval.copy()
        locedge = numpy.zeros_like(f)
        locedge[0,:] = 1
        locedge[:,0] = 1
        locedge[-1,:] = -1
        locedge[:,-1] = -1
        
        extrema_indices = numpy.nonzero(extremaval1)
        tailing_edges = numpy.argwhere(locedge = 1)
        leading_edges = numpy.argwhere(locedge = -1)
        
        
        tailing_edge_values = []
        
        for each in range(len(tailing_edges[0])):
            if tailing_edges[each,0] = 0 and tailing_edges[each,1] = 0:
                tailing_edge_values.append(numpy.mean(x[:1,:1]))
            if tailing_edges[each,0] = 0 and tailing_edges[each,1] != 0:
                tailing_edge_values.append(numpy.mean(x[tailing_edges[each,0],tailing_edges[each,1]:tailing_edges[each,1]+1]))
            if tailing_edges[each,0] != 0 and tailing_edges[each,1] = 0:
                tailing_edge_values.append(numpy.mean(x[tailing_edges[each,0]:tailing_edges[each,0]+1,tailing_edges[each,1]]))
        #get the values for each of the means along the edge        
        
        
        
        
        leading_edge_values = []
        
        for each in range(len(tailing_edges[0])):
            if leading_edges[each,0] = -1 and leading_edges[each,1] = -1:
                leading_edge_values.append(numpy.mean(x[-2:,-2:]))
            if leading_edges[each,0] = -1 and leading_edges[each,1] != -1:
                leading_edge_values.append(numpy.mean(x[tailing_edges[each,0],tailing_edges[each,1]:tailing_edges[each,1]-1]))
            if leading_edges[each,0] != -1 and leading_edges[each,1] = -1:
                leading_edge_values.append(numpy.mean(x[tailing_edges[each,0]:tailing_edges[each,0]-1,tailing_edges[each,1]]))
        #get each of the leading edge mean values
        
        locedge = numpy.zeros_like(f)
        locedge[0,:] = 1
        locedge[:,0] = 1
        locedge[-1,:] = 1
        locedge[:,-1] = 1
        extremaval1 = extremaval + locedge
        extrema_indices = numpy.nonzero(extremaval1)
        edge_indices = numpy.nonzero(locedge)
        
        locedge = numpy.zeros_like(f)
        locedge[0,:] = tailing_edge_values[:locedge[0,:].shape]
        locedge[:,0] = tailing_edge_values[locedge[:,0].shape:]
        locedge[:,-1] = leading_edge_values[locedge[:,0].shape:]
        locedge[-1,:] = leading_edge_values[:locedge[0,:].shape]
        extremaval = extremaval + locedge        
        
        
        
        
        #extrema_indices : all the edges, + the extrema in the array.
        ##extremaval : (hopefully) all the values corresponding to the extrema in extrema_indices
        
        #problem: baseline knots is an array of empty spaces corresponding to the baseline knots, or
        #extrema identified + the edges.
        #however, in 2d, this is a jagged array because there are not the same number of extrema in each row.
        #additionally, there are less extrema than there are edge values.
        #that even assumes my code above is correct, i havn't tested it but it's meant to iterate over each
        #of the edges and store the mean value in a list, and then collect all this together in the
        #correct order.
    
        
        
        
        baseline_knots = numpy.zeros(len(edge_indices) + len(extrema_indices))
        baseline_knots[edge_indices] = extremaval[edge_indices]
        
        #if all of this code would work, in some plausible sense, what it is intended to do
        #is to initialize an array of values corresponding to a number of zeros equivalent to all
        #of the extrema and all of the edges, and then populate the edge values.
        #however, this code will _NOT_ work because i am taking a 2d index to a 1d array-
        #baseline knots is 1d(corresponding to the values in the extrema) the indexes are 2d.
        #the question is how to make this work when it's obviously a jagged array problem.
        
        #furthermore, 
        #what constitutes "before?"
        #how do we interpolate the 1d equations below over 2d?
        #obviously this problem requires integration and other approaches which i am not capable of.
        #i leave these problems to programmers with greater skill than my own.


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

#@numba.jit(numba.float64[:,:](numba.float64[:]))
def itd_2d(data: numpy.ndarray, max_iteration: int = 22) -> numpy.ndarray:
        rotations = numpy.zeros((max_iteration+1,len(data)),dtype=numpy.float64)
        baselines = numpy.zeros((max_iteration+1,len(data)),dtype=numpy.float64)
        rotation_ = numpy.zeros((len(data)),dtype=numpy.float64)
        baseline_ = numpy.zeros((len(data)),dtype=numpy.float64)
        r = numpy.zeros((len(data)),dtype=numpy.float64)
        rotation_[:], baseline_[:] = itd_baseline_extract_2d(numpy.transpose(numpy.asarray(data,dtype=numpy.float64))) 
        counter = 0
        while not finished:         
            idx_max = numpy.asarray(detect_peaks_2d(baseline_))
            idx_min = numpy.asarray(detect_peaks_2d(-baseline_))
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
                rotation_[:],  baseline_[:] = itd_baseline_extract_2d(baseline_[:])
                counter = counter + 1  
