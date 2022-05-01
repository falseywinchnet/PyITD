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



#Interp 1d by Andrei S. Pavlov License: MIT License (MIT)

ITERATION_SOLVER_SIZE_LIMIT = 11
CHECK_INPUT = True

def bisec_find_range(k: float, kp: list):
    k1, k2 = [0, len(kp) - 1]
    while k2 - k1 > 1:
        _ = int((k2 + k1) / 2)
        if k == kp[k1]:
            return k1, k1
        elif k == kp[k2]:
            return k2, k2
        elif kp[k1] < k < kp[_]:
            k2 = _
        elif kp[_] < k < kp[k2]:
            k1 = _
        else:
            return _, _
    return k1, k2

def check_list1d_range(v: float, vp: list):
    if v < vp[0] or v > vp[-1]:
        raise ValueError('value is out of interpolation range')
    if len(vp) < 2:
        raise ValueError('list should have minimum two items')

def check_input1d(x: float, xp: list, yp: list):
    check_list1d_range(x, xp)
    if len(yp) < 2:
        raise ValueError('list should have minimum two items')
    if len(xp) != len(yp):
        raise ValueError('lists should have same length')

def interp1d_bisec(x: float, xp: list, yp: list):
    i1, i2 = bisec_find_range(x, xp)
    return yp[i1] + ((x - xp[i1]) / (xp[i2] - xp[i1])) * (yp[i2] - yp[i1]) if i1 != i2 else yp[i1]

def interp1d_iter(x: float, xp: list, yp: list):
    for ii in range(0, len(xp) - 1):
        if xp[ii] <= x <= xp[ii + 1]:
            return yp[ii] + ((x - xp[ii]) / (xp[ii + 1] - xp[ii])) * (yp[ii + 1] - yp[ii])
    return ValueError('Solution is not find')

def interp1ds(x: float, xp: list, yp: list, make_checks: bool = CHECK_INPUT) -> float:
    if make_checks:
        check_input1d(x, xp, yp)
    return interp1d_bisec(x, xp, yp) if len(xp) > ITERATION_SOLVER_SIZE_LIMIT else interp1d_iter(x, xp, yp)



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

#"""% Matlab Written by Linshan Jia (jialinshan123@126.com)
#% Xi'an Jiaotong University
#% Version 1.0.0
#% 2018-11-04"""

def itd_baseline_extract(data: list[int]) -> (list[int], list[int]):

   #dt = np.dtype([('value', np.float64, 16), ('index', np.int, (2,))])
    x = numpy.zeros_like(data)#   (data.shape, dtype=dt)
    x[:] = numpy.transpose(data[:]) #x=x(:)';
    t = list(range(x.size))
    # t=1:length(x); should do the same as this


    alpha=0.5
    idx_max = detect_peaks(x)


    val_max = x[idx_max] #get peaks based on indexes
    idx_min= detect_peaks(-x)
    val_min = x[idx_min]
    val_min= -val_min

    num_extrema = len(val_max) + len(val_min)# numpy.union1d(idx_max,idx_min)
    extrema_indices = np.zeros((num_extrema + 2), dtype=int)
    extrema_indices[1:-1] = np.union1d(idx_max, idx_min)
    extrema_indices[-1] = len(x) - 1

    baseline_knots = np.zeros(len(extrema_indices))
    baseline_knots[0] = np.mean(x[:2])
    baseline_knots[-1] = np.mean(x[-2:])


    #H = numpy.zeros_like(x)

    #https://localcoder.org/why-does-matlab-interp1-produce-different-results-than-numpy-interp
   # "interp1(x,v,xq) returns interpolated values of a 1-D function at specific query points using linear interpolation.
    # Vector x contains the sample points, and v contains the corresponding values, v(x).
    # Vector xq contains the coordinates of the query points."

    num_extrema = len(val_max) + len(val_min)

    # plt.plot(x[extrema_indices], np.linspace(0, 10, len(extrema_indices)), 'C2.-', lw=0.3, ms=1)
    # plt.xlim(0, 0.2)

    baseline_knots = np.zeros(len(extrema_indices))
    baseline_knots[0] = np.mean(x[:2])
    baseline_knots[-1] = np.mean(x[-2:])

    L = np.zeros_like(x)


    for k in range(1, len(extrema_indices) - 1):
        baseline_knots[k] = alpha * (x[extrema_indices[k - 1]] + \
        (extrema_indices[k] - extrema_indices[k - 1]) / (extrema_indices[k + 1] - extrema_indices[k - 1]) * \
        (x[extrema_indices[k + 1]] - x[extrema_indices[k - 1]])) + \
                            alpha * x[extrema_indices[k]]

    interpolator = interp1d(extrema_indices, baseline_knots / x[extrema_indices], kind='linear')(t)

    Lk1=interpolator[idx_min]+val_min*(1-alpha)
    Lk2=interpolator[idx_max]+val_max*(1-alpha)
   # Lk1=  numpy.concatenate((idx_min, Lk1))  #Lk1=[idx_min(:),Lk1(:)];
  #  Lk2= numpy.concatenate((idx_max,Lk2))

    Lk = np.zeros((num_extrema + 2), dtype=int)
    Lk[1:-1] = numpy.concatenate((Lk1,Lk2))
    Lk[-1] = len(x) - 1

   # Lk=numpy.concatenate((Lk1,Lk2))
    #Lk_col_2 = numpy.argsort(Lk, axis= 0) # sort by first axis?
    #Lk_sorted= Lk[Lk_col_2,:] #Lk_sorted=Lk(Lk_col_2,:);
    #Lk=Lk_sorted[1:-1,:]

    for k in range(len(extrema_indices) - 1):
        for j in range(extrema_indices[k], extrema_indices[k + 1]):
            kij = (Lk[k + 1] - Lk[k]) / (x[extrema_indices[k + 1]] - x[extrema_indices[k]])  # $compute the slope K
            L[j] = Lk[k] + kij * (x[j] - x[extrema_indices[k]])

    H = numpy.subtract(x, L)

    return L,H
def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = numpy.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def itd_baseline_extractr(data: list[int]) -> (list[int], list[int]):

   #dt = np.dtype([('value', np.float64, 16), ('index', np.int, (2,))])
    x = numpy.zeros_like(data)#   (data.shape, dtype=dt)
    x[:] = numpy.transpose(data[:]) #x=x(:)';
    t = list(range(x.size))
    # t=1:length(x); should do the same as this


    alpha=0.5
    idx_max = detect_peaks(x)
    val_max = x[idx_max] #get peaks based on indexes
    idx_min= detect_peaks(-x)
    val_min = x[idx_min]
    val_min= -val_min


    idx_cb = numpy.union1d(idx_max, idx_min)


    print(idx_max.size, val_max.size, idx_min.size, val_min.size)

    if (min(idx_max) < min(idx_min)):
        idx_min = numpy.append(idx_max[0], idx_min[:])
        val_min = numpy.append(val_min[0], val_min[:])

    elif (min(idx_max) > min(idx_min)):
        idx_max = numpy.append(idx_min[0], idx_max[:])
        val_max = numpy.append(val_max[0], val_max[:])

    if (max(idx_max) > max(idx_min)):
        idx_min = numpy.append(idx_min[:], idx_max[-1])
        val_min = numpy.append(val_min[:], val_min[-1])
    elif (max(idx_max) < max(idx_min)):
        idx_max = numpy.append(idx_max[:], idx_min[-1])
        val_max = numpy.append(val_max[:], val_max[-1])


    H = numpy.zeros_like(x)
    L = numpy.zeros_like(x)

    #vq = interp1(x,v,xq) returns interpolated values of a 1-D function at specific query points using linear interpolation.
    #Vector x contains the sample points, and v contains the corresponding values, v(x). Vector xq contains the coordinates of the query points.
    #Max_line = interp1(idx_max, val_max, t, 'linear');


    Max_line = interp1d(idx_max, val_max, kind='linear', bounds_error=False, fill_value="extrapolate")(idx_min)
    Min_line = interp1d(idx_min, val_min, kind='linear', bounds_error=False, fill_value="extrapolate")(idx_max)
    Lk1 = alpha * Max_line + val_min * (1 - alpha)
    Lk2 = alpha * Min_line + val_max * (1 - alpha)


    Lk1=  numpy.hstack((idx_max[:], Lk1[:]))  #Lk1=[idx_min(:),Lk1(:)];
    Lk2= numpy.hstack((idx_min[:],  Lk2[:]))

    Lk = numpy.vstack((Lk1,Lk2))
    Lk_col_2 = numpy.argsort(Lk,axis=1) #an_array[numpy.argsort(an_array[:, 0])]
   #confident above here that we've matched the code
    Lk_sorted= numpy.asarray(np.take_along_axis(Lk, Lk_col_2, axis=1))
    Lk=numpy.delete(Lk_sorted,-1,1)
    Lk=numpy.delete(Lk,0,1)

    rd = numpy.append([1],Lk[0:1])
    vc = numpy.append(len(x), Lk[-1:2])


    Lk=numpy.vstack((Lk[0:1],Lk[0],Lk[1],Lk[-1:2]))
    print(Lk.shape)

    idx_Xk = numpy.concatenate(([1], idx_cb, [x.size]))  # idx_Xk=[1,idx_cb,length(x)];

    for k in range(len(idx_Xk) - 1):
        for j in range(idx_Xk[k], idx_Xk[k + 1]):
            kij = (Lk[k + 1] - Lk[k]) / (x[idx_Xk[k + 1]] - x[idx_Xk[k]])  # $compute the slope K
            L[j] = Lk[k] + kij * (x[j] - x[idx_Xk[k]])
#
    H = numpy.subtract(x, L)

    return L,H



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
        audio = (1.0*(self.buffer - np.min(self.buffer))/np.ptp(self.buffer)).astype(numpy.float64)
        #normalize inputs
        origin = audio[:, 0]
        results = ITD(audio[:, 0])
        results = numpy.squeeze(numpy.asarray(results))
        results = numpy.swapaxes(results,0,1)
        comparison = numpy.sum(results, axis=1)
        comparison = comparison * 2.0
        sumc = numpy.sum(comparison)
        sumd = numpy.sum(audio)
        print(sumc,sumd)
        #numpy.flip(results)
        #x = numpy.ravel(results, order='C')
        #self.processedrb.write(self.buffer.astype(dtype=self.dtype), error=True) #UNCOMMENT ALSO PLAY
        #Z, freqs, t = mlab.specgram(x, NFFT=256, Fs=len(x), detrend=None, window=None, noverlap=223, pad_to=None, scale_by_freq=None, mode="default")
        c = (255*(results - np.min(results))/np.ptp(results)).astype(int)
        image = c.astype('float64')
        # https://stackoverflow.com/questions/39359693/single-valued-array-to-rgba-array-using-custom-color-map-in-python
        arr_color = self.SM.to_rgba(image, bytes=False, norm=True)
        arr_color = snowy.resize(arr_color, width=60, height=100)  # in the future, this width will be 60.
        arr_color = numpy.rot90(arr_color)  # rotate it and jam it in the buffer lengthwise
        self.cleanspecbuf.growing_write(arr_color)
        #np.set_printoptions(threshold=np.inf, linewidth=200)
        #self.cleanspecbuf.growing_write(arr_color)
        #with open("ITD.txt", "ab") as f:
        #    f.write(b"\n")
        #    numpy.savetxt(f, results, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)


        return

        #for i in range(self.channels):
            # do work on ITD here
        #self.iterations = iterationz
        #self.processedrb.write(self.buffer2.astype(dtype=self.dtype), error=True)

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

    def __init__(self, sample_rate=8000, channels=2, buffer_delay=1.5,  # or 1.5, measured in seconds
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
        if True: # len(self.processedrb) < self.processing_size:
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
            discard = SS.cleanspectrogrambuffer.read(1) #sneakily throw away rows until we're back to sanity




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
