from typing import Iterable

import pyfftw
import planfftw
import numpy as np

# copied from the matlab by falsy winchnet and wealthybanker57
# Emperical Fourier Decomposition technique
# uses fftw, pyfftw, planfftw for fast and easy precise decomp accuracy ~1.0-14 error
# which is around the precision FFT is capable of

def segm_tec(f, N: int):
    # detect the local maxima
    locmax = np.zeros_like(f)
    arg_maxima = np.argwhere(np.r_[True, f[1:] > f[:-1]] & np.r_[f[:-1] > f[1:], True])
    locmax[arg_maxima] = f[arg_maxima]

    # detect local minima (not too sure why these are unused in the matlab code,
    # will leave them commented for future use)
#     locmin = np.ones_like(f) * np.max(f)
#     arg_minima = np.argwhere(np.r_[True, f[1:] < f[:-1]] & np.r_[f[:-1] < f[1:], True])
#     locmin[arg_minima] = f[arg_minima]

    locmax[0] = f[0]
    locmax[-1] = f[-1]

    if N != 0: # keep the N-th highest maxima and their index
        desc_sort_index = locmax.argsort()[::-1]
        desc_sort = locmax[desc_sort_index]

        desc_sort_index = np.sort(desc_sort_index[0:N])
        N = len(desc_sort_index)
        M = N + 1 # numbers of the boundaries
        omega = np.concatenate(([0], desc_sort_index))
        omega = np.concatenate((omega, [len(f)]))
        bounds = np.zeros((M))
        for i in range(M):
            if (i == 0 or i == M) and (omega[i] == omega[i+1]):
                bounds[i] = omega[i] - 1
            else:
                ind = np.argmin(f[omega[i]:omega[i+1]])
                bounds[i] = omega[i] + ind - 2
        cerf = desc_sort_index * np.pi / len(f)
    return bounds, cerf


# https://arxiv.org/pdf/2009.08047v2.pdf
def EFD(x: Iterable[np.float64], N: int):
    # x is the signal, N is the number of maxima to use in computations
    
    # we will now implement the Empirical Fourier Decomposition
    # what is it with me and signal decomposition approaches??
    x = np.asarray(x, dtype=np.float64)
    # we will assume that x is 1d, if x is 2d, test and transform to put rows-first

    fx = planfftw.fft(x)
    ff = fx(x)
    half_ff = len(ff) // 2

    # extract the boundaries of Fourier segments
    bounds, cerf = segm_tec(abs(ff[0:half_ff]), N)
    bounds = np.concatenate(([0], bounds)) #fix : 5/10/22 temp patch for the first value
    # truncate the boundaries to [0, pi]
    bounds = bounds * np.pi / half_ff

    # extend the signal by miroring to deal with the boundaries
    l = len(x) // 2
    # x = [x(l-1:-1:1);x;x(end:-1:end-l+1)];
    z = np.concatenate((np.flip(x[:l]), x))
    z = np.concatenate((z, np.flip(x[-l:])))

    fr = planfftw.fft(z)
    ff = fr(z)

    # obtain the boundaries in the extend f
    bound2 = np.ceil(bounds * half_ff / np.pi).astype(dtype=int)
    # bound2 = np.concatenate((bound2,[8000]))
    efd = np.zeros(((len(bound2) - 1, len(x))),dtype=np.float64)
    ft = np.zeros((efd.shape[0], len(ff)), dtype=np.cdouble)
    fz = planfftw.ifft(ft[0, :])
    # define an ideal functions and extract components
    for k in range(efd.shape[0]):
        reused_bound = len(ff) + 2 - bound2[k+1]
        if bound2[k] == 0:
            ft[k, 0:bound2[k+1]] = ff[0:bound2[k+1]]
            ft[k, reused_bound:len(ff)] = ff[reused_bound:len(ff)]
        else:
            ft[k, bound2[k]:bound2[k+1]] = ff[bound2[k]:bound2[k+1]]
            ft[k, reused_bound:len(ff) + 2 - bound2[k]] = ff[reused_bound:len(ff) + 2 - bound2[k]]
        rx = np.real(fz(ft[k, :]))
        efd[k, :] = rx[l: -l]

    return efd, cerf, bounds
