import numpy
#copied from the matlab by falsy winchnet.
#Emperical Fourier Decomposition technique

def segm_tec(f, N):
    locmax = numpy.zeros((f.size))
    for i in range(1, len(f) - 1):
        if ((f[i - 1] <= f[i]) and (f[i] > f[i + 1])):
            if (f[i] > 0):  # a value below 0 or 0 is not a maxima!
                locmax[i] = f[i]

    desc_sort_index = numpy.argsort(locmax)[::-1]
    desc_sort_bool = numpy.empty(desc_sort_index.shape[0], dtype=bool)
    for i in range(desc_sort_index.size):
        if locmax[i] > 0:
            desc_sort_bool[i] = True

    desc_sort_index = desc_sort_index[desc_sort_bool]

    if N != 0:  # keep the N-th highest maxima and their index
        if len(desc_sort_index) > N:
            desc_sort_index = desc_sort_index[0:N + 1]
        else:
            N = desc_sort_index.size
        desc_sort_index = numpy.sort(desc_sort_index)  # gotta sort them again
        bounds = numpy.empty(N+2, dtype=int)
        bounds[0] = 0
        bounds[1] = (numpy.argmin(f[0:desc_sort_index[0]]))  # -2
        for i in range(N - 2):
            bounds[i+2] = (desc_sort_index[i] + numpy.argmin(f[desc_sort_index[i]:desc_sort_index[i+1]]) - 1)
        bounds[-2] = (desc_sort_index[N] + numpy.argmin(f[desc_sort_index[N]:len(f)]) - 1)
        bounds[-1] = f.size
        cerf = desc_sort_index * numpy.pi / round(len(f))
    return numpy.asarray(bounds), cerf

#https://arxiv.org/pdf/2009.08047v2.pdf
def EFD(x: list[numpy.float64], N: int):
    #we will now implement the Empirical Fourier Decomposition
    x = numpy.asarray(x,dtype=numpy.float64)
    ff = numpy.fft.rfft(x)
    #extract the boundaries of Fourier segments
    bounds,cerf = segm_tec(abs(ff[0:round(ff.size/2)]),N)
    # truncate the boundaries to [0,pi]
    bounds = bounds*numpy.pi/round(len(ff)/2)
    
    # extend the signal by miroring to deal with the boundaries
    l = round(len(x)/2)
    z = numpy.lib.pad(x,((round(len(x)/2)),round(len(x)/2)),'symmetric') 
    ff =  numpy.fft.rfft(z)
    # obtain the boundaries in the extend f
    bound2 = numpy.ceil(bounds*round(len(ff)/2)/numpy.pi).astype(dtype=int)
    efd = numpy.zeros(((len(bound2)-1,len(x))),dtype=numpy.float64)
    ft = numpy.zeros((efd.shape[0],len(ff)),dtype=numpy.cdouble)
    # define an ideal functions and extract components
    for k in range(efd.shape[0]): 
        if bound2[k] == 0:
            ft[k,0:bound2[k+1]] = ff[0:bound2[k+1]]
            ft[k,-bound2[k+1]:len(ff)] = ff[-bound2[k+1]:len(ff)]
        else:
            ft[k,bound2[k]:bound2[k+1]] = ff[bound2[k]:bound2[k+1]]
            ft[k,-bound2[k+1]:-bound2[k]] = ff[-bound2[k+1]:-bound2[k]]
        rx = numpy.fft.irfft(ft[k,:])
        efd[k,:] = rx[l:-l]

    return efd,cerf,bounds
