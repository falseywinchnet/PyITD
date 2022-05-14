import pyfftw
import planfftw
#warning: the results of this output are not proven in any sense, way, shape, or form.
#you use this at your own peril.

def segm_tec2d(f,N):
# 1. detect the local maxima and minina
    locmax = numpy.zeros_like(f)
    locmin = numpy.ones_like(f)
    locmin = locmin * numpy.max(f)
    for i in range(1,f.shape[0]-1):#iterate over columns 
        for j in range(1,f.shape[1]-1): #iterate over rows
            if (f[i-1,j-1] < f[i,j]) and (f[i,j]>f[i+1,j+1]) and \
             (f[i-1,j] < f[i,j]) and (f[i,j]>f[i+1,j]) and \
             (f[i,j-1] < f[i,j]) and (f[i,j]>f[i,j+1]) and \
                (f[i+1,j-1] < f[i,j]) and (f[i,j] >f[i+1,j-1]) and \
                (f[i-1,j+1] < f[i,j]): #consider all 9 neighbors
                locmax[i,j]=f[i,j]  
            #if (f[i-1,j-1] > f[i,j]) and (f[i,j] < f[i+1,j+1]) and \
            # (f[i-1,j] > f[i,j]) and (f[i,j] < f[i+1,j]) and \
            # (f[i,j-1] > f[i,j]) and (f[i,j] < f[i,j+1]) and \
            #    (f[i+1,j-1] > f[i,j]) and (f[i,j]< f[i+1,j-1]) and \
            #    (f[i-1,j+1] > f[i,j]): #consider all 9 neighbors
            #    locmin[i,j]=f[i,j] # This equation is not used?
    locmax[0,:] = f[0,:]
    locmax[:,0] = f[:,0]
    locmax[-1,:] = f[-1,:]
    locmax[:,-1] = f[:,-1] #establish the framework
                  
    if N != 0: #keep the N-th highest maxima and their index
        desc_sort = -numpy.sort(-locmax)#perform a descending sort
        desc_sort_index = locmax.argsort()[::-1]
        
        if len(desc_sort) > N:
            desc_sort_index = numpy.sort(desc_sort_index[0:N,0:N])
        else:
            desc_sort_index = numpy.sort(desc_sort_index)
            N = len(desc_sort)
        M = N+1# numbers of the boundaries
        
        omega =  desc_sort_index.copy()
        omega = numpy.insert(omega, 0, 0, axis=0)
        omega = numpy.insert(omega, 0, 0, axis=1)
        omega = numpy.insert(omega, omega.shape[0], f.shape[0], axis=0)
        omega = numpy.insert(omega, omega.shape[1], f.shape[1], axis=1)
        #elegant way to prepend 0s and append 1s to 2d array    
        
        bounds = numpy.zeros((M,M))
        for i in range(M):
            for j in range(M):
                if (i == 0 or i == M) and (omega[i,j] == omega[i+1,j+1]) \
                    and (omega[i,j] == omega[i,j+1]) and (omega[i,j] == omega[i+1,j]) or \
                    (j == 0 or j == M) and (omega[i,j] == omega[i+1,j+1]) \
                    and (omega[i,j] == omega[i,j+1]) and (omega[i,j] == omega[i+1,j]):
                    bounds[i,j] = omega[i,j]-1
                else:
                    #what do we do about 
                    ind = numpy.argmin(f[omega[i:i+1,j:j+1]])
                    bounds[i,j] = omega[i,j]+ind-2;
        cerf = desc_sort_index*numpy.pi/round(len(f))
    return bounds, cerf


#https://arxiv.org/pdf/2009.08047v2.pdf
def EFD2d(x: list[numpy.float64], N: int):
    #we will now attempt 2d EFD
    #what is it with me and signal decomposition approaches??
    x = numpy.asarray(x,dtype=numpy.float64)
    #we will assume that x is 1d, if x is 2d, test and transform to put rows-first
    
    fx =  planfftw.fftn(x)
    ff = fx(x)

    #extract the boundaries of Fourier segments
    bounds,cerf = segm_tec2d(abs(ff[0:round(ff.size/2)]),N)
    
    bounds =  numpy.insert(bounds, 0, 0, axis=0)
    bounds =  numpy.insert(bounds, 0, 0, axis=1)#prepend zeros
    # truncate the boundaries to [0,pi]
    bounds = bounds*numpy.pi/round(ff[0,:].size/2)
    
    
    # extend the signal by miroring to deal with the boundaries
    l = round(x[0,:].size/2)
    #x = [x(l-1:-1:1);x;x(end:-1:end-l+1)];
    
    z = x.copy()
    z = numpy.lib.pad(z,((l,l),(l,l)),'reflect') 

    
    fr =  planfftw.fftn(z)
    ff = fr(z)

    # obtain the boundaries in the extend f
    bound2 = numpy.ceil(bounds*round(ff[0,:].size/2)/numpy.pi).astype(dtype=int)  
    
    efd = numpy.zeros(((bound2[0,:].size-1,bound2[:,0].size-1, x[0,:].size, x[:,0].size)),dtype=numpy.float64)
    #generalize EFD to a 3d vector constrained by K?
    
    ft = numpy.zeros((bound2[0,:].size-1,bound2[:,0].size-1,ff[0,:].size,ff[:,0].size),dtype=numpy.cdouble)
    
    fz =  planfftw.ifftn(ft[0,0,:,:]) 
    
    
    
    
    for k in range(bound2[0,:].size-1): 
        for j in range(bound2[:,0].size-1):
            if bound2[k,j] == 0:
                ft[k,j,0:bound2[k+1,j+1]] = ff[0:bound2[k+1,j+1]]
                ft[k,j,ff[0,:].size+2-bound2[k+1,j+1]:len(ff)] = ff[ff[0,:].size +2-bound2[k+1,j+1]:ff[0,:].size]
            else:
                ft[k,j,bound2[k,j]:bound2[k+1,j+1]] = ff[bound2[k,j]:bound2[k+1,j+1]]
                ft[k,j,ff[0,:].size+2-bound2[k+1,j+1]:ff[0,:].size+2-bound2[k,j]]\
                = ff[ff[0,:].size+2-bound2[k+1,j+1]:ff[0,:].size+2-bound2[k,j]]
            rx = numpy.real(fz(ft[k,j,:,:]))
            efd[k,j,:,:] = rx[l:-l,l:-l]


    return efd,cerf,bounds
