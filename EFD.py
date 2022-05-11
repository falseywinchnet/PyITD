import pyfftw
import planfftw
#copied from the matlab by falsy winchnet.
#Emperical Fourier Decomposition technique
#uses fftw, pyfftw, planfftw for fast and easy precise decomp accuracy ~1.0-14 error
#which is around the precision FFT is capable of

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
    locmax[0] = f[0];
    locmax[-1] = f[-1];
    print(len(f))
                  
    if N != 0: #keep the N-th highest maxima and their index
        desc_sort = -numpy.sort(-locmax)#perform a descending sort
        desc_sort_index = locmax.argsort()[::-1]
        
        if len(desc_sort) > N:
            desc_sort_index = numpy.sort(desc_sort_index[0:N])
        else:
            desc_sort_index = numpy.sort(desc_sort_index)
            N = len(desc_sort)
        M = N+1# numbers of the boundaries
        omega = numpy.concatenate(([0],desc_sort_index))
        omega = numpy.concatenate((omega,[len(f)]))
        bounds = numpy.zeros((M))
        for i in range(M):
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
    #what is it with me and signal decomposition approaches??
    x = numpy.asarray(x,dtype=numpy.float64)
    #we will assume that x is 1d, if x is 2d, test and transform to put rows-first
    
    fx =  planfftw.fft(x)
    ff = fx(x)

    #extract the boundaries of Fourier segments
    bounds,cerf = segm_tec(abs(ff[0:round(ff.size/2)]),N)
    bounds = numpy.concatenate(([0],bounds)) #fix : 5/10/22 temp patch for the first value
    # truncate the boundaries to [0,pi]
    bounds = bounds*numpy.pi/round(len(ff)/2)
    

    # extend the signal by miroring to deal with the boundaries
    l = round(len(x)/2)
    #x = [x(l-1:-1:1);x;x(end:-1:end-l+1)];
    z = numpy.concatenate((numpy.flip(x[:l]),x))
    z = numpy.concatenate((z,numpy.flip(x[-l:])))

    fr =  planfftw.fft(z)
    ff = fr(z)

    # obtain the boundaries in the extend f
    bound2 = numpy.ceil(bounds*round(len(ff)/2)/numpy.pi).astype(dtype=int)
    #bound2 = numpy.concatenate((bound2,[8000]))
    efd = numpy.zeros(((len(bound2)-1,len(x))),dtype=numpy.float64)
    ft = numpy.zeros((efd.shape[0],len(ff)),dtype=numpy.cdouble)
    fz =  planfftw.ifft(ft[0,:])
    # define an ideal functions and extract components
    for k in range(efd.shape[0]): 
        if bound2[k] == 0:
            ft[k,0:bound2[k+1]] = ff[0:bound2[k+1]]
            ft[k,len(ff)+2-bound2[k+1]:len(ff)] = ff[len(ff)+2-bound2[k+1]:len(ff)]
        else:
            ft[k,bound2[k]:bound2[k+1]] = ff[bound2[k]:bound2[k+1]]
            ft[k,len(ff)+2-bound2[k+1]:len(ff)+2-bound2[k]] = ff[len(ff)+2-bound2[k+1]:len(ff)+2-bound2[k]]
        rx = numpy.real(fz(ft[k,:]))
        print(rx.shape)
        efd[k,:] = rx[l:-l]


    return efd,cerf,bounds
