import pyfftw
import planfftw
import numpy
#copied from the matlab by falsy winchnet.
#Emperical Fourier Decomposition technique


def segm_tec(f,N):
    locmax = numpy.zeros_like(f)
    for i in range(1,len(f)-1): 
        if ((f[i-1] <= f[i]) and (f[i] > f[i+1])):
            if(f[i] > 0): #a value below 0 or 0 is not a maxima!
                locmax[i]= f[i]
    
    desc_sort_index = numpy.argsort(locmax)[::-1]
    count_nonzeros = numpy.nonzero(locmax)
    desc_sort_bool = numpy.isin(desc_sort_index,count_nonzeros)# get the top amplitudes which are peaks
    desc_sort_index = desc_sort_index[desc_sort_bool]
    
    #ok, now we have our maxima
    
    if N != 0: #keep the N-th highest maxima and their index
        if len(desc_sort_index) > N:
            desc_sort_index = desc_sort_index[0:N+1]
        else:
            N = desc_sort_index.size
        desc_sort_index = numpy.sort(desc_sort_index) #gotta sort them again 
        bounds = []
        bounds.append(0)
        bounds.append(numpy.argmin(f[0:desc_sort_index[0]])) # -2
        for i in range(1,N - 1):
            bounds.append(desc_sort_index[i-1] + numpy.argmin(f[desc_sort_index[i-1]:desc_sort_index[i]]) - 1)     
        bounds.append(desc_sort_index[N] + numpy.argmin(f[desc_sort_index[N]:len(f)]) - 1)
        bounds.append(len(f))
        bounds = numpy.asarray(bounds)
        cerf = desc_sort_index*numpy.pi/round(len(f))
    return bounds, cerf


#https://arxiv.org/pdf/2009.08047v2.pdf
def EFD(x: list[numpy.float64], N: int):
    #we will now implement the Empirical Fourier Decomposition
    x = numpy.asarray(x,dtype=numpy.float64)
    
    fx =  planfftw.fft(x)
    ff = fx(x)
    #extract the boundaries of Fourier segments
    bounds,cerf = segm_tec(abs(ff[0:round(ff.size/2)]),N)
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
    efd = numpy.zeros(((len(bound2)-1,len(x))),dtype=numpy.float64)
    ft = numpy.zeros((efd.shape[0],len(ff)),dtype=numpy.cdouble)
    fz =  planfftw.ifft(ft[0,:])
    # define an ideal functions and extract components
    for k in range(efd.shape[0]): 
        if bound2[k] == 0:
            ft[k,0:bound2[k+1]] = ff[0:bound2[k+1]]
            ft[k,len(ff)+1-bound2[k+1]:len(ff)] = ff[len(ff)+1-bound2[k+1]:len(ff)]
        else:
            ft[k,bound2[k]:bound2[k+1]] = ff[bound2[k]:bound2[k+1]]
            ft[k,len(ff)+1-bound2[k+1]:len(ff)+1-bound2[k]] = ff[len(ff)+1-bound2[k+1]:len(ff)+1-bound2[k]]
        rx = numpy.real(fz(ft[k,:]))
        efd[k,:] = rx[l:-l]


    return efd,cerf,bounds
