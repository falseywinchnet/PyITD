#modified emperical fourier composition, by falseywinchnet

import numpy
import numba


@numba.njit(numba.boolean[:](numba.int64[:],numba.int64[:]))
def isin(a, b):
    out=numpy.empty(a.shape[0], dtype=numba.boolean)
    b = set(b)
    for i in range(a.shape[0]):
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

@numba.jit(numba.types.Tuple((numba.int64[:],numba.int64,numba.int64[:]))(numba.float64[:],numba.int64))
def segm_tec(f:numpy.ndarray, N:int):
    zbounds = numpy.zeros((N + 3), dtype=numpy.int64)
    x = numpy.asarray(f)
    
    ind =  detect_peaks(-x)
    if ind.size < 4:
        return zbounds, 0, zbounds

    ind = ind[ind>0] #end cannot be peak
    ind = ind[ind<x.size] #end cannot be peak
    if ind.size < 2:
        return zbounds, 0, zbounds

    locmax = numpy.zeros((ind.size),dtype=numpy.float64)
    locmax[:] = x[ind]

    desc_sort = numpy.argsort(locmax)[::-1]
    sorted = ind[desc_sort]

    if N == 1:
        desc_sort_indice = sorted[0]
    else:
        if N < sorted.size:
            desc_sort_index = sorted[0:N] 
            N = desc_sort_index.size

        else:
          desc_sort_index = sorted
        desc_sort_index = numpy.sort(desc_sort_index)  # gotta sort them again
        N = desc_sort_index.size

    bounds = numpy.zeros((N + 3), dtype=numpy.int64)


    if N == 1: #if only one peak is desired
        bounds[1] = (numpy.argmin(x[0:desc_sort_indice]))
        bounds[2] = (desc_sort_indice + numpy.argmin(x[desc_sort_indice:x.size]))
    else:
        bounds[1] = (numpy.argmin(x[0:desc_sort_index[0]]))  

        for i in range(N - 1):
            bounds[i + 2] = (desc_sort_index[i] + numpy.argmin(x[desc_sort_index[i]:desc_sort_index[i + 1]]))
        bounds[-2] = (desc_sort_index[-1] + numpy.argmin(x[desc_sort_index[-1]:x.size]))
    bounds[-1] = x.size
    
    return numpy.asarray(bounds),N, numpy.argsort(x[desc_sort_index])[::-1]



import numpy as np

def  EFD_real(row, elem):
    robust = np.fft.irfft(row)
    bounds,N,sort = segm_tec(robust[0:robust.size//2],elem)
    if(N!=elem):
      print("warning, only peaks found")
      print(N)
      elem = N
    result = []
    z = numpy.zeros(len(robust))
    for i in range(elem+2):
        z[:] = 0.0
        z[bounds[i]:bounds[i+1]] = robust[bounds[i]:bounds[i+1]]
        z[-bounds[i+1]:-bounds[i]] = robust[-bounds[i+1]:-bounds[i]]
        working = numpy.fft.rfft(z).real
        result.append(working)


    return result,sort

def  iterative(data, elem,comb_size):
  working = data.copy()
  result = []
  for each in range(elem):
    first,sort = EFD_real(working,comb_size)
    result.append(first[sort[0]+1])#+1 because we start with 0-argmin
    working = working - first[sort[0]+1]
  result.append(working)
  return result

#this method attempts to find the top impulse out of comb_size partitions and iteratively extract it.
#this method tends to be computationally intensive. therefore, every effort has been made to accelerate it.
#unfortunantly, this still requires a lot of FFT operations for a complete decomposition

def  EFD_slice_max(row, elem):
    robust = np.fft.irfft(row)
    bounds,N,sort = segm_tec(robust[0:robust.size//2],elem)
    if(N==0):
      print("warning, no peaks found")
      return row,sort
    if(N!=elem):
      print("warning, only peaks found")
      print(N)
      elem = N
    result = []
    z = numpy.zeros(len(robust))
    z[bounds[sort[0]+1]:bounds[sort[0]+2]] = robust[bounds[sort[0]+1]:bounds[sort[0]+2]]
    z[-bounds[sort[0]+2]:-bounds[sort[0]+1]] = robust[-bounds[sort[0]+2]:-bounds[sort[0]+1]]
    working = numpy.fft.rfft(z).real

    return working

def  iterative_max(row, elem,comb_size):
  working = row.copy()
  result = []
  for each in range(elem):
    first = EFD_slice_max(working,comb_size)
    result.append(first)#+1 because we start with 0-argmin
    working = working - first
  result.append(working)
  return result

#this result should require less and allow more specific selection of impulses
