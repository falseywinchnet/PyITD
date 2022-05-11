#Author: falseywinchnet, shamelessly copied from matlab code and adapted for numpy.
#warning : this code does not generate good output yet. Further validation is needed.





def segm_tec(f,N):
# 1. detect the local maxima and minina
    locmax = numpy.zeros_like(f);
    locmin = numpy.ones_like(f);
    locmin = locmin * numpy.max(f)
    for i in range(1,len(f)-1): 
        if ((f[i-1]<f[i]) and (f[i]>f[i+1])):
            locmax[i]=f[i]
        if ((f[i-1]>f[i]) and (f[i]<f[i+1])):
            locmin[i]=f[i]
    locmax[1] = f[1];
    locmax[-1] = f[-1];
    
                  
    if N != 0: #keep the N-th highest maxima and their index
        desc_sort = -numpy.sort(-locmax)#perform a descending sort
        desc_sort_index = locmax.argsort()[::-1]
        
        if len(desc_sort) > N:
            desc_sort_index = numpy.sort(desc_sort_index[0:N])
        else:
            desc_sort_index = numpy.sort(desc_sort_index)
            N = len(desc_sort)
        M = N+1# numbers of the boundaries
        omega = numpy.concatenate(([1],desc_sort_index))
        omega = numpy.concatenate((desc_sort_index,[len(f)]))
        bounds = numpy.zeros((M))
        for i in range(M-1):
            if (i == 0 or i == M) and (omega[i] == omega[i+1]):
                bounds[i] = omega[i]-1;
            else:
                ind = numpy.argmin(f[omega[i]:omega[i+1]])
                bounds[i] = omega[i]+ind-2;
        cerf = desc_sort_index*numpy.pi/round(len(f))
    return bounds, cerf


#https://arxiv.org/pdf/2009.08047v2.pdf
def EFD(x: list[numpy.float64], N: int):
    #we will now implement the Empirical Fourier Decomposition

    #we will assume that x is 1d, if x is 2d, test and transform to put rows-first
    ff = numpy.fft.fft(x)

    #extract the boundaries of Fourier segments
    bounds,cerf = segm_tec(abs(ff[0:round(ff.size/2)]),N)

    # truncate the boundaries to [0,pi]
    bounds = bounds*numpy.pi/round(len(ff)/2)
    

    # extend the signal by miroring to deal with the boundaries
    l = round(len(x)/2)
    #x = [x(l-1:-1:1);x;x(end:-1:end-l+1)];
    z = numpy.concatenate((numpy.flip(x[0:l-1]),x))
    z = numpy.concatenate((z,numpy.flip(x[l+1:])))
    ff = numpy.fft.fft(z)

    # obtain the boundaries in the extend f
    bound2 = numpy.ceil(bounds*round(len(ff)/2)/numpy.pi).astype(dtype=int)
    print(bound2)
    efd = numpy.zeros(((len(bound2)-1,len(x))))
    ft = numpy.zeros((efd.shape[0],len(ff)),dtype=numpy.cdouble)
    
    # define an ideal functions and extract components
    for k in range(efd.shape[0]): 
        if bound2[k] == 0:
            ft[k,1:bound2[k+1]] = ff[1:bound2[k+1]]
            ft[k,len(ff)+2-bound2[k+1]:len(ff)] = ff[len(ff)+2-bound2[k+1]:len(ff)]
        else:
            ft[k,bound2[k]:bound2[k+1]] = ff[bound2[k]:bound2[k+1]]
            ft[k,len(ff)+2-bound2[k+1]:len(ff)+2-bound2[k]] = ff[len(ff)+2-bound2[k+1]:len(ff)+2-bound2[k]]
        rx = numpy.real(numpy.fft.ifft(ft[k,:]))
        efd[k,:] = rx[l:-l+2]


    return efd,cerf,bounds


