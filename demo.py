"""

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.


Instructions:
Save the code as a .pyw file to disable the console. Save as a .py to enable it.
Install the latest miniforge for you into a folder, don't add it to path, launch it's command line from start menu.
Note: if python is installed elsewhere this may fail. If it fails, try this again with miniconda instead,
as miniconda's pip doesn't install packages to the system library locations when python is installed.

https://github.com/conda-forge/miniforge/#download

(using miniforge command line window)

 conda create --name fabada --no-default-packages python=3.10
 conda activate fabada
 pip install pipwin, dearpygui, numba, np_rw_buffer,matplotlib, snowy
 pipwin install pyaudio


pythonw.exe thepythonfilename.py #assuming the python file is in the current directory


Usage:
You'll need a line-in device or virtual audio cable you can configure so you can loop the output to input.
The streaming example included here looks for the windows muxer that configures audio devices- whatever you set
in your windows settings for default mic and speaker, respectively, this program will treat as input and output.


Step one: configure sound as follows : output device is speakers, i nput device is virtual cable
step two using app volume and system preferences, route your desired output to the speaker side of the cable
step three Start the python program

https://vb-audio.com/Cable/ is an example of a free audio cable.
The program expects 44100hz audio, 16 bit, two channel, but can be configured to work with anything
Additional thanks to Justin Engel.
ITD implementation knot and trend fixing code by Chromum
"""

from __future__ import division,print_function

import numpy
import numpy as np
import pyaudio
#import numba
from matplotlib import mlab
from np_rw_buffer import AudioFramingBuffer
from np_rw_buffer import RingBuffer
from threading import Thread
import math
import time
import array
from time import sleep
import dearpygui.dearpygui as dpg
import snowy
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import scipy
import awkward as ak
import numba
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges, TestSignals





@numba.jit(numba.float64[:](numba.float64[:], numba.int32, numba.float64[:]), nopython=True, parallel=True, nogil=True, cache=True)
def shift1d(arr : list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:num] = fill_value[:num]
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value[:num]
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

@numba.jit(numba.float64[:,:](numba.float64[:,:], numba.int32, numba.float64[:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift2dy(arr: list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:,:num] = fill_value[:,:num]
        result[:,num:] = arr[:,:-num]
    elif num < 0:
        result[:,num:] = fill_value[:,:num]
        result[:,:num] = arr[:,-num:]
    else:
        result[::] = arr
    return result

@numba.jit(numba.float64[:,:,:](numba.float64[:,:,:], numba.int32, numba.float64[:,:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift3dx(arr: list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:num,:,:] = fill_value[:num,:,:]
        result[num:,:,:] = arr[:-num,:,:]
    elif num < 0:
        result[num:,:,:] = fill_value[:num,:,:]
        result[:num,:,:] = arr[-num:,:,:]
    else:
        result[:] = arr
    return result

@numba.jit(numba.float32[:,:,:](numba.float32[:,:,:], numba.int32, numba.float32[:,:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift3dximg(arr: list[numpy.float32], num: int, fill_value: list[numpy.float32]) -> list[numpy.float32] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:,:num,:] = fill_value
        result[:,num:,:] = arr[:,:-num,:]
    elif num < 0:
        result[:,num:,:] = fill_value
        result[:,:num,:] = arr[:,-num:,:]
    else:
        result[:] = arr
    return result
#because numpy's zeroth array is the Y axis, we have to do this in the 1st dimension to shift the X axis
#if the arrays are not the same size, don't attempt to use coordinates for fill value- it will fail.


ITERATION_SOLVER_SIZE_LIMIT = 11
CHECK_INPUT = True

#numba doesnt support numpy.isin yet
@numba.njit(parallel=True)
def isin(a, b):
    out=np.empty(a.shape[0], dtype=numba.boolean)
    b = set(b)
    for i in numba.prange(a.shape[0]):
        if a[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out


@numba.jit(numba.float64[:](numba.float64[:]),parallel=True,cache=True)
def numba_fabada(data: [numpy.float64]):
        #notes:
        #The pythonic way to COPY an array is to do x[:] = y[:]
        #do x=y and it wont copy it, so any changes made to X will also be made to Y.
        #also, += does an append instead of a +
        #math.sqrt is 7x faster than numpy.sqrt but not designed for complex numbers.
        #specifying the type and size in advance of all variables accelerates their use, particularily when JIT is used.
        #However, JIT does not permit arbitrary types like UNION? maybe it does but i havnt figured out how.
        #this implementation uses a lot of for loops because they can be easily vectorized by simply replacing
        #range with numba.prange and also because they will translate well to other languages
        #this implementation of FABADA is not optimized for 2d arrays, however, it is easily swapped by changing the means
        #estimation and by simply changing all other code to iterate over 2d instead of 1d
        #care must be taken with numba parallelization/vectorization
        with numba.objmode(start=numba.float64):
            start = time.time()

        iterations: int = 1
        TAU: numpy.float64 = 2 * math.pi
        N = data.size

        #must establish zeros for the model or otherwise when data is empty, algorithm will return noise
        bayesian_weight  = numpy.zeros_like(data)
        bayesian_model = numpy.zeros_like(data)
        model_weight = numpy.zeros_like(data)

        #pre-declaring all arrays allows their memory to be allocated in advance
        posterior_mean  = numpy.empty_like(data)
        posterior_variance  = numpy.empty_like(data)
        initial_evidence = numpy.empty_like(data)
        evidence = numpy.empty_like(data)
        prior_mean = numpy.empty_like(data)
        prior_variance = numpy.empty_like(data)

        #working set arrays, no real meaning, just to have work space
        ja1 = numpy.empty_like(data)
        ja2 = numpy.empty_like(data)
        ja3 = numpy.empty_like(data)
        ja4 = numpy.empty_like(data)

        #eliminate divide by zero
        data[data == 0.0] = 2.22044604925e-16
        min_d: numpy.float64 = numpy.min(data)
        max_d: numpy.float64 = numpy.ptp(data)
        min: numpy.float64 = 2.22044604925e-16
        max:numpy.float64 =  44100
        #the higher max is, the less "crackle".
        #The lower floor is, the less noise reduction is possible.
        #floor can never be less than max/2.
        #noise components are generally higher frequency.
        #the higher the floor is set, the more attenuation of both noise and signal.


        posterior_mean[:] = data[:]
        prior_mean[:] = data[:]
        data_mean = numpy.mean(data)  # get the mean
        data_variance = numpy.empty_like(data)

        for i in numba.prange(N):
            data_variance[i] = numpy.abs(data_mean - max_d) ** 2

        posterior_variance[:] = data_variance[:]

        for i in numba.prange(N):
            ja1[i] = ((0.0 - math.sqrt(data[i])) ** 2)
            ja2[i] = ((0.0 + data_variance[i]) * 2)
            ja3[i] = math.sqrt(TAU * (0.0 + data_variance[i]))
        for i in numba.prange(N):
            ja4[i] = math.exp(-ja1[i] / ja2[i])
        for i in numba.prange(N):
            evidence[i] = ja4[i] / ja3[i]
        evidence_previous: numpy.float64 = numpy.mean(evidence)
        initial_evidence[:] = evidence[:]

        for i in numba.prange(N):
            ja1[i] = data[i] - posterior_mean[i]
        for i in numba.prange(N):
            ja1[i] = ja1[i] ** 2.0 / data_variance[i]
        chi2_data_min: numpy.float64 = numpy.sum(ja1)
        chi2_pdf_previous: numpy.float64 = 0.0
        chi2_pdf_derivative_previous: numpy.float64 = 0.0
        # COMBINE MODELS FOR THE ESTIMATION


        while 1:

        # GENERATES PRIORS
            prior_mean[:] = posterior_mean[:]
            prior_mean[:-1] += posterior_mean[1:]
            prior_mean[1:] += posterior_mean[:-1]
            prior_mean[1:-1] /= 3
            prior_mean[0] /= 2
            prior_mean[-1] /= 2

            # APPLY BAYES' THEOREM ((b\a)a)\b?
            prior_variance[:] = posterior_variance[:]

            for i in numba.prange(N):
                posterior_variance[i] = 1.0 / (1.0/ data_variance[i] + 1.0 / prior_variance[i])
            for i in numba.prange(N):
                posterior_mean[i] = (((prior_mean[i] / prior_variance[i]) + ( data[i] / data_variance[i])) * posterior_variance[i])

            # EVALUATE EVIDENCE

            for i in numba.prange(N):
                ja1[i] = ((prior_mean[i] - math.sqrt(data[i])) ** 2)
                ja2[i] = ((prior_variance[i] + data_variance[i]) * 2)
                ja3[i] = math.sqrt(TAU * (prior_variance[i] + data_variance[i]))
            for i in numba.prange(N):
                ja4[i] = math.exp(-ja1[i] / ja2[i])
            for i in numba.prange(N):
                evidence[i] = ja4[i] / ja3[i]

            evidence_derivative: numpy.float64 = numpy.mean(evidence) - evidence_previous

            # EVALUATE CHI2

            for i in numba.prange(N):
                ja1[i] = ((data[i] - posterior_mean[i]) ** 2 / data_variance[i])
            chi2_data = numpy.sum(ja1)


            # COMBINE MODELS FOR THE ESTIMATION
            for i in numba.prange(N):
                model_weight[i] = evidence[i] * chi2_data

            for i in numba.prange(N):
                bayesian_weight[i] = bayesian_weight[i] + model_weight[i]
                bayesian_model[i] = bayesian_model[i] + (model_weight[i] * posterior_mean[i])

            df: int = 5  # note: this isnt the right way to use this function. DF is supposed to be, like data.size but that would be enormous
            #for any data set which is non-trivial. Remember, DF/2 - 1 becomes the exponent! For anything over a few hundred this quickly exceeds float64.
            ## chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
            gammar: numpy.float64 = (2. * math.lgamma(df / 2.))
            gammaz: numpy.float64 = ((df / 2.) - 1.)
            gamman: numpy.float64 = (chi2_data / 2.)
            gammas: numpy.float64 = (numpy.sign(gamman) * (
                        (abs(gamman)) ** gammaz))  #TODO
            if math.isnan(gammas):
                gammas = (numpy.sign(gamman) * ((abs(gamman)) * gammaz))
            gammaq: numpy.float64 = math.exp(-chi2_data / 2.)
            #for particularily large values, math.exp just returns 0.0
            #TODO
            gammaa: numpy.float64 = 1. / gammar
            chi2_pdf = gammaa * gammas * gammaq

            # COMBINE MODELS FOR THE ESTIMATION

            chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
            chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous
            chi2_pdf_previous = chi2_pdf
            chi2_pdf_derivative_previous = chi2_pdf_derivative
            evidence_previous: numpy.float64 = evidence_derivative

            with numba.objmode(current=numba.float64):
                current = time.time()
            timerun = (current - start) * 1000

            if (
                    (int(chi2_data) > data.size and chi2_pdf_snd_derivative >= 0)
                    or (abs(evidence_derivative) < 0)
                    or (iterations > 100)  # use no more than 95% of the time allocated per cycle
            ):
                break

            iterations += 1


            # COMBINE ITERATION ZERO
        for i in numba.prange(N):
            model_weight[i] = initial_evidence[i] * chi2_data_min
        for i in numba.prange(N):
            bayesian_weight[i] = (bayesian_weight[i]  + model_weight[i])
            bayesian_model[i] = bayesian_model[i] + ( model_weight[i] *  data[i])

        for i in numba.prange(N):
            data[i] = bayesian_model[i] / bayesian_weight[i]

        return data


@numba.jit(numba.int64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
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



#@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def ITD(data: list[int]):
    # notes:
    # The pythonic way to COPY an array is to do x[:] = y[:]
    # do x=y and it wont copy it, so any changes made to X will also be made to Y.
    # also, += does an append instead of a +
    # math.sqrt is 7x faster than numpy.sqrt but not designed for complex numbers.
    # specifying the type and size in advance of all variables accelerates their use, particularily when JIT is used.
    # However, JIT does not permit arbitrary types like UNION? maybe it does but i havnt figured out how.
    # this implementation uses a lot of for loops because they can be easily vectorized by simply replacing
    # range with numba.prange and also because they will translate well to other languages
    # this implementation of FABADA is not optimized for 2d arrays, however, it is easily swapped by changing the means
    # estimation and by simply changing all other code to iterate over 2d instead of 1d
    # care must be taken with numba parallelization/vectorization
    #we will now implement the intrinsic time-scale decomposition algorithm.
    #function H=itd(x)
    N_max = 10
    working_set = numpy.zeros_like(data)
    working_set[:] = data[:]
    xx = working_set.transpose()
    E_x = sum(numpy.square(working_set)) # same thing as E_x=sum(x.^2);
    counter = 0
    STOP = False
    #we have to initialize the array, because awkward doesn't have an empty array initializer-
    #and because we do not know the first value! of the array
    counter = counter + 1
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

@numba.jit
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
#https://github.com/numba/numba/issues/4119
@numba.jit(numba.float64[:](numba.float64[:],numba.float64[:]),parallel=True,nogil=True,cache=True)
def numba_convolve_mode_valid_as_loop(arr, kernel):
    m = arr.size
    n = kernel.size
    out_size = m - n + 1
    out = numpy.empty(out_size, dtype=numpy.float64)
    for i in numba.prange(out_size):
        out[i] = numpy.dot(arr[i:i + n], kernel)
    return out

@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True, cache=True)
def savgol(data: list[numpy.float64]):
    coeff = numpy.asarray([-0.08571429, 0.34285714, 0.48571429, 0.34285714, -0.08571429])
    # pad_length = h * (width - 1) // 2# for width of 5, this defaults to 2...
    data_pad = numpy.zeros(data.size + 4)
    data_pad[2:data.size + 2] = data[0:data.size]  # leaving two on each side
    firstval = 2 * data[0] - data[2:0:-1]
    lastvals = 2 * data[-1] - data[-1:-3:-1]
    data_pad[0] = firstval[0]
    data_pad[1] = firstval[1]
    data_pad[-1] = lastvals[1]
    data_pad[-2] = lastvals[0]
    new_data = numpy.zeros((data.size))

    #
    # multiply vec2 by vec1[0] = 2    4   6
    # multiply vec2 by vec1[1] = -    3   6   9
    # multiply vec2 by vec1[2] = -    -   4   8   12
    # -----------------------------------------------
    # add the above three      = 2    7   16  17  12

    new_data[:] = numba_convolve_mode_valid_as_loop(data_pad[:], coeff[:])

    # create the array of each set of averaging values

    return new_data

def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = numpy.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


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
    #y_interp = np.interp(x_interp, x, y) yields an interpolation of the function y_interp = f(x_interp)
    # based on a previous interpolation y = f(x), where x.size = y.size, x_interp.size = y_interp.size.
    #scipy is (idx_min,val_min)(t)[idx_max]
    #interpolator = interp1d(extrema_indices, baseline_knots / x[extrema_indices], kind='linear')(t)

    #x = np.interp(y_max, y_data, x_data,  left=None, right=None, period=None)
    #max_line = numpy.interp(max(max_knots),max_knots, idx_max,  left=None, right=None, period=None)[t]


    num_extrema = len(val_max) + len(val_min)# numpy.union1d(idx_max,idx_min)
    extrema_indices = np.zeros(((num_extrema + 2)), dtype=numpy.int)
    extrema_indices[1:-1] = np.union1d(idx_max, idx_min)
    extrema_indices[-1] = len(x) - 1

    baseline_knots = np.zeros(len(extrema_indices))
    baseline_knots[0] = np.mean(x[:2])
    baseline_knots[-1] = np.mean(x[-2:])


    baseline_knots = np.zeros(len(extrema_indices))
    baseline_knots[0] = np.mean(x[:2])
    baseline_knots[-1] = np.mean(x[-2:])

    for k in range(1, len(extrema_indices) - 1):
        baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] + \
        (extrema_indices[k] - extrema_indices[k - 1]) / (extrema_indices[k + 1] - extrema_indices[k - 1]) * \
        (x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]])) + \
                            alpha * x[extrema_indices[k]]

    interpolator = numpy.interp(t,extrema_indices, baseline_knots / x[extrema_indices])

    #print(interpolator_numba,interpolator,len(interpolator), len(interpolator_numba))
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
#
    H = numpy.subtract(x, L)

    return L,H
import numpy as np
from numba import njit

@numba.njit(fastmath=True)
def trilinear_interpolation_jit(
    x_volume,
    y_volume,
    z_volume,
    volume,
    x_needed,
    y_needed,
    z_needed
):
    """
    Trilinear interpolation (from Wikipedia)

    :param x_volume: x points of the volume grid
    :type crack_type: list or numpy.ndarray
    :param y_volume: y points of the volume grid
    :type crack_type: list or numpy.ndarray
    :param x_volume: z points of the volume grid
    :type crack_type: list or numpy.ndarray
    :param volume:   volume
    :type crack_type: list or numpy.ndarray
    :param x_needed: desired x coordinate of volume
    :type crack_type: float
    :param y_needed: desired y coordinate of volume
    :type crack_type: float
    :param z_needed: desired z coordinate of volume
    :type crack_type: float

    :return volume_needed: desired value of the volume, i.e. volume(x_needed, y_needed, z_needed)
    :type volume_needed: float
    :author Pietro D'Antuono
    """

    # dimensinoal check
    assert np.shape(volume) == (
        len(x_volume), len(y_volume), len(z_volume)
    ), "Incompatible lengths"
    # check of the indices needed for the correct control volume definition
    i = np.searchsorted(x_volume, x_needed)
    j = np.searchsorted(y_volume, y_needed)
    k = np.searchsorted(z_volume, z_needed)
    # control volume definition
    control_volume_coordinates = np.array(
        [
            [
                x_volume[i - 1],
                y_volume[j - 1],
                z_volume[k - 1]
            ],
            [
                x_volume[i],
                y_volume[j],
                z_volume[k]
            ]
        ]
    )
    xd = (
        np.array([x_needed, y_needed, z_needed]) - control_volume_coordinates[0]
    ) / (
        control_volume_coordinates[1] - control_volume_coordinates[0]
    )
    # interpolation along x
    c2 = [[0., 0.], [0., 0.]]
    for m, n in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        c2[m][n] = volume[i - 1][j - 1 + m][k - 1 + n] \
        * (1. - xd[0]) + volume[i][j - 1 + m][k - 1 + n] * xd[0]
    # interpolation along y
    c1 = [0., 0.]
    c1[0] = c2[0][0] * (1. - xd[1]) + c2[1][0] * xd[1]
    c1[1] = c2[0][1] * (1. - xd[1]) + c2[1][1] * xd[1]
    # interpolation along z
    volume_needed = c1[0] * (1. - xd[2]) + c1[1] * xd[2]
    return volume_needed

@njit(fastmath=True)
def trilint_jit(
    x_volume,
    y_volume,
    z_volume,
    volume,
    x_needed,
    y_needed,
    z_needed
):
    trilint_size = x_needed.size * y_needed.size * z_needed.size
    jitted_trilint = np.zeros(trilint_size)
    m = 0
    for x in range(0, len(x_needed)):
        for y in range(0, len(y_needed)):
            for z in range(0, len(z_needed)):
                jitted_trilint[m]=trilinear_interpolation_jit(
                    x_volume,
                    y_volume,
                    z_volume,
                    volume,
                    x_needed[x],
                    y_needed[y],
                    z_needed[z]
                )
                m = m + 1
    return jitted_trilint


@numba.jit(numba.float64[:,:](numba.float64[:]))
def decomposeinto3d(input: list[numpy.float64]):
    #here's where the magic happens.
    # decompose it into a three dimensional graph
    #time corresponds to sample position in x.
    #the elements are also sorted by frequency, and then the array is transposed
    #and the position of each corresponds to frequency in another dimension.
    #in a third dimension, energy corresponds to amplitude.
    output = numpy.ndarray(input.size, input.size, numpy.ptp(input))  # create the terrain array
    unique_elements, frequency = np.unique(input, return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    #sorted_by_freq = unique_elements[sorted_indexes]
    output[:,0] = range(len(input))
    for each in output[:,1]:
        output[each,sorted_indexes[each]] = input[each]#place the amplitude at the frequency point
                                                        #to rerieve them, all we have to do later is reverse this.

    return output


@numba.jit(numba.float64[:,:,:](numba.float64[:,:,:]))
def denoise3d(input: list[numpy.float64]):
        #apply terrain walking methods to interpolate all results.
        #we apply guassian smoothing to this terrain(energy minima).
        #first dimension is index values
        #ie
       # 1 2 3 4 5 6 7 8 9 0
        #our second dimension is sorted indexes- frequency
       # ie
        #1 22 23 24 40 90
        #our third dimension is amplitude
       # ie
      #  1.3 5.3 20.1 305 503 30

        #our first dimension is x
        #our second dimension is y
       # our third dimension is z
        #input = [:,:]x, y where amplitude is placed at y(frequency) and time(x)
        #as long as different decomopositional trends don't include multiple amplitudes at the same frequency and time
        #- by definition this *should* be impossible- then this will result in a 3d graph of values.

        x_volume = np.array([100., 1000.])
        y_volume = np.array([0.2, 0.4, 0.6, 0.8, 1])
        z_volume = np.array([0, 0.2, 0.5, 0.8, 1.])
        x_needed = np.linspace(100, 1000, 10)
        y_needed = np.linspace(0.3, 1, 60)
        z_needed = np.linspace(0, 1, 7)
        jitted_trilint = trilint_jit(
            x_volume, y_volume, z_volume, input, x_needed, y_needed, z_needed
        )

        #incorporate terrain smoothing guassian filter here
        output = 1
        return output



class FilterRun(Thread):
    def __init__(self, rb, pb, channels, processing_size, dtype,work,time,floor,iterations,clean,run,f):
        super(FilterRun, self).__init__()
        self.running = True
        self.rb = rb
        self.processedrb = pb
        self.channels = channels
        self.processing_size = processing_size
        self.dtype = dtype
        self.buffer = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = self.buffer.reshape(-1, self.channels)
        self.buffer = self.buffer.reshape(-1, self.channels)
        self.work = work
        self.time = time
        self.floor = floor
        self.iterations = iterations
        self.cleanspecbuf = clean
        self.enabled = run
        self.NFFT = 512
        self.noverlap=446
        self.SM = cm.ScalarMappable(cmap="turbo")
        self.last = [0.,0.]
        self.f = f





    def write_filtered_data(self):

        numpy.copyto(self.buffer, self.rb.read(self.processing_size).astype(dtype=numpy.float64))
        min_d: numpy.float64 = numpy.min(self.buffer[:, 0])
        max_d: numpy.float64 = numpy.ptp(self.buffer[:, 0])
        #audio = numpy.interp(self.buffer[:, 0], (min_d, max_d),(min, max))#normalize the data
        audio = ((self.buffer[:, 0] - min_d) / (max_d - min_d)).astype(numpy.float64)
        #normalize inputs

        results = ITD(audio)
        results[-1,:] = savgol(results[-1,:]) #smooth the trend? not sure if this is needed or desired
        result3d=numpy.ndarray((results[:,1].size,results[1,:].size)) #create the terrain array
        for i in (results[:,1].size - 1):
            result3d[i,:,:] = decomposeinto3d(results[i,:])
        result3d = result3d.sum(axis=0)#generate the three dimensional representation
        result3d = denoise3d(result3d)
        resultsmooth = result3d[2,:]#decompose the terrain by simply time and amplitude
        results = resultsmooth + results[-1,:] #recompose the data

        results1 = (results * (max_d - min_d)) + min_d #denormalize the data
        self.buffer2[:, 0] = results1
        self.buffer2[:, 1] = self.buffer[:, 1]

        Z, freqs, t = mlab.specgram(results1, NFFT=256, Fs=44100, detrend=None, window=None, noverlap=223, pad_to=None, scale_by_freq=None, mode="default")
        ##c = (255*(results - np.min(results))/np.ptp(results)).astype(int)
        #image = c.astype('float64')
        # https://stackoverflow.com/questions/39359693/single-valued-array-to-rgba-array-using-custom-color-map-in-python
        arr_color = self.SM.to_rgba(Z, bytes=False, norm=True)
        arr_color = arr_color[1:40,-50:,:]
        arr_color = snowy.resize(arr_color, width=60, height=100)
        arr_color = numpy.rot90(arr_color)  # rotate it and jam it in the buffer lengthwise
        self.cleanspecbuf.growing_write(arr_color)
        self.processedrb.write(self.buffer2.astype(dtype=self.dtype), error=True)

        #np.set_printoptions(threshold=np.inf, linewidth=200)
        #self.cleanspecbuf.growing_write(arr_color)
        #with open("ITD.txt", "ab") as f:
         #   f.write(b"\n\n")
         #   r1 = numpy.flip(results)
         #   numpy.savetxt(f, r1, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)


        return

        #for i in range(self.channels):
            # do work on ITD here
        #self.iterations = iterationz
        #self.processedrb.write(results1(dtype=self.dtype), error=True)

        #Z, freqs, t = mlab.specgram(self.buffer2[:, 0], NFFT=256, Fs=44100, detrend=None, window=None, noverlap=223,
                          #          pad_to=None, scale_by_freq=None, mode="magnitude")

        # https://stackoverflow.com/questions/39359693/single-valued-array-to-rgba-array-using-custom-color-map-in-python


    def run(self):


        while self.running:
            if len(self.rb) < self.processing_size * 2:
                sleep(0.05)  # idk how long we should sleep
            else:
                self.write_filtered_data()

    def stop(self):
        self.running = False


class StreamSampler(object):
    dtype_to_paformat = {
        # Numpy dtype : pyaudio enum
        'uint8': pyaudio.paUInt8,
        'int8': pyaudio.paInt8,
        'uint16': pyaudio.paInt16,
        'int16': pyaudio.paInt16,
        'uint24': pyaudio.paInt24,
        'int24': pyaudio.paInt24,
        "uint32": pyaudio.paInt32,
        'int32': pyaudio.paInt32,
        'float32': pyaudio.paFloat32,

        # Float64 is not a valid pyaudio type.
        # The encode method changes this to a float32 before sending to audio
        'float64': pyaudio.paFloat32,
        "complex128": pyaudio.paFloat32,
    }

    @classmethod
    def get_pa_format(cls, dtype):
        try:
            dtype = dtype.dtype
        except (AttributeError, Exception):
            pass
        return cls.dtype_to_paformat[dtype.name]

    def __init__(self, sample_rate=44100, channels=2, buffer_delay=1.5,  # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.float32):
        #22050 is a common sampling rate for data analysis
        self.pa = pyaudio.PyAudio()
        self._processing_size = sample_rate
        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        self.ticker = 0
        # self.rb = RingBuffer((int(sample_rate) * 5, channels), dtype=numpy.dtype(dtype))
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=8,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=0,  # #this buffer doesnt need to have a size
                                     dtype=numpy.dtype(dtype))

        self.processedrb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                              seconds=8,
                                              # Buffer size (need larger than processing size)[seconds * sample_rate]
                                              buffer_delay=1,
                                              dtype=numpy.dtype(dtype))
        self.cleanspectrogrambuffer = RingBuffer((660, 100, 4),dtype=numpy.float32)
        self.cleanspectrogrambuffer.maxsize = int(9900)
        self.texture2 = [1., 1., 1., 1.] * 500 * 100
        self.texture2 = numpy.asarray( self.texture2,dtype=numpy.float32)
        self.texture2 = self.texture2.reshape((100, 500, 4)) #create and shape the textures. Backwards.
        self.work = 1. #included only for completeness
        self.time = 495 #generally, set this to whatever timeframe you want it done in. 44100 samples = 500ms window.
        self.floor = 8192#unknown, seems to do better with higher values
        self.iterations = 0
        self.enabled = False
        self.f = open('test.txt', 'w')
        self.filterthread = FilterRun(self.rb, self.processedrb, self._channels, self._processing_size, self.dtype,self.work,self.time, self.floor,self.iterations,self.cleanspectrogrambuffer,self.enabled,self.f)
        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None
        self.speakerdevice = ""
        self.micdevice = ""



        # Set inputs for inheritance
        self.set_sample_rate(sample_rate)
        self.set_channels(channels)
        self.set_dtype(dtype)

    @property
    def processing_size(self):
        return self._processing_size

    @processing_size.setter
    def processing_size(self, value):
        self._processing_size = value
        self._update_streams()

    def get_sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, value):
        self._sample_rate = value
        try:  # RingBuffer
            self.rb.maxsize = int(value * 5)
            self.processedrb.maxsize = int(value * 5)
        except AttributeError:
            pass
        try:  # AudioFramingBuffer
            self.rb.sample_rate = value
            self.processedrb.sample_rate = value
        except AttributeError:
            pass
        self._update_streams()

    sample_rate = property(get_sample_rate, set_sample_rate)

    def get_channels(self):
        return self._channels

    def set_channels(self, value):
        self._channels = value
        try:  # RingBuffer
            self.rb.columns = value
            self.processedrb.columns = value
        except AttributeError:
            pass
        try:  # AudioFrammingBuffer
            self.rb.channels = value
            self.processedrb.channels = value
        except AttributeError:
            pass
        self._update_streams()

    channels = property(get_channels, set_channels)

    def get_dtype(self):
        return self.rb.dtype

    def set_dtype(self, value):
        try:
            self.rb.dtype = value
        except AttributeError:
            pass
        self._update_streams()

    dtype = property(get_dtype, set_dtype)

    @property
    def pa_format(self):
        return self.get_pa_format(self.dtype)

    @pa_format.setter
    def pa_format(self, value):
        for np_dtype, pa_fmt in self.dtype_to_paformat.items():
            if value == pa_fmt:
                self.dtype = numpy.dtype(np_dtype)
                return

        raise ValueError('Invalid pyaudio format given!')

    @property
    def buffer_delay(self):
        try:
            return self.rb.buffer_delay
        except (AttributeError, Exception):
            return 0

    @buffer_delay.setter
    def buffer_delay(self, value):
        try:
            self.rb.buffer_delay = value
            self.processedrb.buffer_delay = value
        except AttributeError:
            pass

    def _update_streams(self):
        """
Call if sample
rate, channels, dtype, or something
about
the
stream
changes.
"""
        was_running = self.is_running()

        self.stop()
        self.micstream = None
        self.speakerstream = None
        if was_running:
            self.listen()

    def is_running(self):
        try:
            return self.micstream.is_active() or self.speakerstream.is_active()
        except (AttributeError, Exception):
            return False

    def stop(self):
        try:
            self.micstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.speakerstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.filterthread.join()
        except (AttributeError, Exception):
            pass

    def open_mic_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxInputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.micdevice = devinfo["name"]
                        device_index = i
                        self.micindex = device_index

        if device_index is None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              # each frame carries twice the data of the frames
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_read,
                              start=False  # Need start to be False if you don't want this to start right away
                              )

        return stream

    def open_speaker_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxOutputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.speakerdevice = devinfo["name"]
                        device_index = i
                        self.speakerindex = device_index

        if device_index is None:
            print("No preferred output found; using default output device.")

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_write,
                              start=False  # Need start to be False if you don't want this to start right away
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    # attention: numpy.ndarray is actually faster than frombuffer for known buffer sizes
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = memoryview(numpy.ndarray(buffer=memoryview(in_data), dtype=self.dtype,
                                            shape=[int(self._processing_size * self._channels)]).reshape(-1,
                                                                                                         self.channels))
        self.rb.write(audio_in, error=False)
        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        # Read raw data
        # filtered = self.rb.read(frame_count)
        # if len(filtered) < frame_count:
        #     filtered = numpy.zeros((frame_count, self.channels), dtype=self.dtype)
        if len(self.processedrb) < self.processing_size:
            # print('Not enough data to play! Increase the buffer_delay')
            # uncomment this for debug
            audio = numpy.zeros((self.processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue
        #SKIP ALL THE CODE BELOW HERE IN DEBUG
        audio = self.processedrb.read(self.processing_size)
        chans = []
        for i in range(self.channels):
            filtered = audio[:, i]
            chans.append(filtered)

        return numpy.column_stack(chans).astype(self.dtype).tobytes(), pyaudio.paContinue



    def stream_start(self):
        if self.micstream is None:
            self.micstream = self.open_mic_stream()
        self.micstream.start_stream()

        if self.speakerstream is None:
            self.speakerstream = self.open_speaker_stream()
        self.speakerstream.start_stream()
        # Don't do this here. Do it in main. Other things may want to run while this stream is running
        # while self.micstream.is_active():
        #     eval(input("main thread is now paused"))

    listen = stream_start  # Just set a new variable to the same method


texture_data = []
for i in range(0, 500 * 100):
    texture_data.append(255 / 255)
    texture_data.append(0)
    texture_data.append(255 / 255)
    texture_data.append(255 / 255)

    # patch from joviex- the enumeration in the online docs showing .append doesn't work for larger textures

raw_data2 = array.array('f', texture_data)
    #declare globals here. These are universally accessible.

if __name__ == "__main__":
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    SS = StreamSampler(buffer_delay=0)
    SS.listen()
    SS.filterthread.start()
    def close():
        dpg.destroy_context()
        SS.filterthread.stop()
        SS.stop()
        quit()



    def update_spectrogram_textures():
        # new_color = implement buffer read
        if len(SS.cleanspectrogrambuffer) > 60:
            SS.texture2 = shift3dximg(SS.texture2, -1, numpy.rot90(SS.cleanspectrogrambuffer.read(1), 1))
        if len(SS.cleanspectrogrambuffer) > 180: #clearly this is wrooong
            discard = SS.cleanspectrogrambuffer.read(10) #sneakily throw away rows until we're back to sanity




    def iter():
        update_spectrogram_textures() #update the screen contents once every frame
        dpg.set_value("clean_texture", SS.texture2)


    dpg.create_context()
    dpg.create_viewport(title='ITD Demo', height=300, width=500)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)




    with dpg.texture_registry():
        dpg.add_raw_texture(500, 100, raw_data2, format=dpg.mvFormat_Float_rgba, tag="clean_texture")


    with dpg.window(height = 300, width = 500) as main_window:
        dpg.add_text("Welcome to DEMO! 1S delay typical")
        dpg.add_text(f"Your speaker device is: ({SS.speakerdevice})")
        dpg.add_text(f"Your microphone device is:({SS.micdevice})")
        dpg.add_image("clean_texture")

    dpg.set_primary_window(main_window,True)  # TODO: Added Primary window, so the dpg window fills the whole Viewport

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        iter()#this runs once a frame.
        dpg.render_dearpygui_frame()
    close() #clean up the program runtime when the user closes the window
