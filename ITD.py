#License: None, use at your own peril. If you want to give credit, falseywinchnet ported, @Chronum94 contributed akima spline interpolation,
# Linshan Jia (jialinshan123 at 126) wrote the matlab original code, frei and osorio wrote the original algorithm, and 
#https://arxiv.org/pdf/1404.3827v1.pdf the authors of this paper wrote an algorithm representation on page 26 which i cant read so i havnt validated
#Intrinsic Time-Scale Representation Algorithm

import numpy
import numpy as np


def detect_peaks(x: list[numpy.float64]):
    """Detect peaks in data based on their amplitude and other features.
    warning: this code is an optimized copy of the "Marcos Duarte, https://github.com/demotu/BMC"
    matlab compliant detect peaks function intended for use with data sets that only want
    rising edge and is optimized for numba. experiment with it at your own peril.
    """
    # find indexes of all peaks
    x = numpy.asarray(x)
    if len(x) < 3:
        return np.empty(1, np.int64)
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    indl = numpy.asarray(indnan)

    if indl.size!= 0:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    vil = numpy.zeros(dx.size + 1)
    vil[:-1] = dx[:]# hacky solution because numba does not like hstack tuple arrays
    #np.asarray((dx[:], [0.]))# hacky solution because numba does not like hstack
    vix = numpy.zeros(dx.size + 1)
    vix[1:] = dx[:]

    ind = numpy.unique(np.where((vil <= 0) & (vix > 0))[0])
    # handle NaN's
    # NaN's and values close to NaN's cannot be peaks
    if ind.size and indl.size:
        outliers = np.unique(np.concatenate((indnan, indnan - 1, indnan + 1)))
        booloutliers = isin(ind, outliers)
        booloutliers = numpy.invert(booloutliers)
        ind = ind[booloutliers]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    return ind



def ITD(data: list[int]):
    # notes:
    # The python way to COPY an array is to do x[:] = y[:]
    # do x=y and it wont copy it, so any changes made to X will also be made to Y.
    # also, += does an append instead of a = a +
    # specifying the type and size in advance of all arrays accelerates their use, particularily when JIT is used.

    N_max = 10
    working_set = numpy.zeros_like(data)
    working_set[:] = data[:]
    xx = working_set.transpose()
    E_x = sum(numpy.square(working_set)) # same thing as E_x=sum(x.^2);
    STOP = False
    counter = 1
    Lx, Hx = itd_baseline_extract(xx)
    L1 = numpy.asarray(Lx)
    H = numpy.asarray(Hx)
    STOP = stop_iter(xx, counter, N_max, E_x)
    if STOP:
        print("finished in one iteration")
        return H

    xx = numpy.asarray(L1)

    while 1:
        counter = counter + 1
        Lx, Hx = itd_baseline_extract(xx)
        L1= numpy.asarray(Lx)
        H = numpy.vstack((H,numpy.asarray(Hx)))

        STOP = stop_iter(xx, counter, N_max, E_x)
        if STOP:
            print("reached stop in ", counter, " iterations.")
            H = numpy.vstack((H, numpy.asarray(L1)))
            break
        xx = numpy.asarray(L1)
    return H

def stop_iter(xx,counter,N_max,E_x) -> (bool):
    if (counter>N_max):
        return True
    Exx= sum(numpy.square(xx))
    exr = 0.01 * E_x
    truth = numpy.less_equal(Exx,exr)
    if truth:
       print("value exceeded truth")
       return True
    #https://blog.ytotech.com/2015/11/01/findpeaks-in-python/ we may want to switch
    #to the PeakUtils interpolate function for better results
    #however, since there is no filtering going on here, we will use Marcos Duarte's code
    pks1= set(detect_peaks(xx))
    pks2= set(detect_peaks(-xx))

    pks= pks1.union(pks2)
    if (len(pks)<=7):
        return True

    return False


def itd_baseline_extract(data: list[int]) -> (list[int], list[int]):

   #dt = np.dtype([('value', np.float64, 16), ('index', np.int, (2,))])
    x = numpy.asarray(numpy.transpose(data[:])) #x=x(:)';
    t = list(range(x.size))
    # t=1:length(x); should do the same as this

    alpha=0.5
    idx_max = detect_peaks(x)
    val_max = x[idx_max] #get peaks based on indexes
    idx_min= detect_peaks(-x)
    val_min = x[idx_min]
    val_min= -val_min

    H = numpy.zeros_like(x)
    L = numpy.zeros_like(x)

    num_extrema = len(val_max) + len(val_min)# numpy.union1d(idx_max,idx_min)
    extrema_indices = np.zeros(((num_extrema + 2)), dtype=numpy.int)
    extrema_indices[1:-1] = np.union1d(idx_max, idx_min)
    extrema_indices[-1] = len(x) - 1

    baseline_knots = np.zeros(len(extrema_indices))
    baseline_knots[0] = np.mean(x[:2])
    baseline_knots[-1] = np.mean(x[-2:])

    for k in range(1, len(extrema_indices) - 1):
        baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] + \
        (extrema_indices[k] - extrema_indices[k - 1]) / (extrema_indices[k + 1] - extrema_indices[k - 1]) * \
        (x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]])) + \
                            alpha * x[extrema_indices[k]]

    interpolator = numpy.interp(t,extrema_indices, baseline_knots / x[extrema_indices])

    Lk1 = np.asarray(alpha * interpolator[idx_min] + val_min * (1 - alpha))
    Lk2 = np.asarray(alpha * interpolator[idx_max] + val_max * (1 - alpha))

    Lk1 = numpy.hstack((np.atleast_2d(idx_min).T, np.atleast_2d(Lk1).T))
    Lk2 = numpy.hstack((np.atleast_2d(idx_max).T, np.atleast_2d(Lk2).T))
    Lk = numpy.vstack((Lk1,Lk2))
    Lk = Lk[Lk[:,1].argsort()]
    if Lk.size > 6:
        Lk = Lk[1:-1,:]

    Ls = numpy.asarray(([1],Lk[0,1]))
    Lk = numpy.vstack((Ls,Lk))
    Ls = numpy.asarray(([len(x)], Lk[-1, 1]))
    Lk = numpy.vstack((Lk, Ls))

    idx_Xk = numpy.concatenate(([0], extrema_indices, [x.size]))  # idx_Xk=[1,idx_cb,length(x)];
    for k in range(len(idx_Xk) - 5):
        for j in range(idx_Xk[k], idx_Xk[k + 1]):
            vk = (Lk[k + 1, 1] - Lk[k,1])
            sk = (x[idx_Xk[k + 1]] - x[idx_Xk[k]])
            kij = vk / sk  # $compute the slope K
            L[j] = Lk[k,1] + kij * (x[j] - x[idx_Xk[k]])
            
    H = numpy.subtract(x, L)

    return L,H
