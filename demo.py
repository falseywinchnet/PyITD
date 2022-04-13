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
#import snowy
import matplotlib.cm as cm
from scipy.interpolate import interp1d



#@numba.jit(numba.float64[:](numba.float64[:], numba.int32, numba.float64[:]), nopython=True, parallel=True, nogil=True, cache=True)
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

#@numba.jit(numba.float64[:,:](numba.float64[:,:], numba.int32, numba.float64[:,:]), nopython=True, parallel=True, nogil=True,cache=True)
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

#@numba.jit(numba.float64[:,:,:](numba.float64[:,:,:], numba.int32, numba.float64[:,:,:]), nopython=True, parallel=True, nogil=True,cache=True)
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

#@numba.jit(numba.float32[:,:,:](numba.float32[:,:,:], numba.int32, numba.float32[:,:,:]), nopython=True, parallel=True, nogil=True,cache=True)
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



def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's 
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()




#@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def ITD(data: list[numpy.float64]) -> ( list[numpy.float64]):
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
    N_max = 10;
    working_set = numpy.zeros_like(data)
    working_set[:] = data[:]

    H = []
    L1 = []
    H1 = numpy.empty_like(data)
    xx = working_set.transpose()
    E_x = numpy.square(working_set) # same thing as E_x=sum(x.^2);
    counter = 0
    STOP = False
    while 1:
        counter = counter + 1
        L1, H1 = itd_baseline_extract(xx)
        H = numpy.vstack(H, H1)
        STOP = stop_iter(xx, counter, N_max, E_x)
        if STOP:
            H = numpy.vstack(H, L1)
            break
        xx[:] = L1[:]

    return xx



def stop_iter(xx,counter,N_max,E_x) -> (bool):
    if (counter>N_max):
        return True

    Exx=numpy.square(xx)
    if (Exx<=0.01 * E_x):
        return True
    #https://blog.ytotech.com/2015/11/01/findpeaks-in-python/ we may want to switch
    #to the PeakUtils interpolate function for better results
    #however, since there is no filtering going on here, we will use Marcos Duarte's code
    pks1= detect_peaks(xx)
    pks2= detect_peaks(-xx)
    pks= set.union(pks1, pks2)
    if (len(pks)<=7):
        return True
    return False

#"""% Matlab Written by Linshan Jia (jialinshan123@126.com)
#% Xi'an Jiaotong University
#% Version 1.0.0
#% 2018-11-04"""


def itd_baseline_extract(data: list[numpy.float64]) -> (list[numpy.float64], list[numpy.float64]):



    x = numpy.transpose(data[:]) #x=x(:)';
    t = list(range(x.size))
    # t=1:length(x); should do the same as this


    alpha=0.5
    idx_max = detect_peaks(x)
    val_max = x[idx_max] #get peaks based on indexes
    idx_min= detect_peaks(-x)
    val_min = x[idx_min]
    idx_cb= numpy.union1d(idx_max,idx_min)
    val_min= -val_min
    if (min(idx_max)<min(idx_min)):
        idx_min = numpy.append(idx_max[0],idx_min[:])
        val_min = numpy.append(val_min[0],val_min[:])

    elif ( min(idx_max)> min(idx_min)):
        idx_max=numpy.append(idx_min[0],idx_max[:])
        val_max=numpy.append(val_max[0],val_max[:])

    if (max(idx_max)>max(idx_min)):
        idx_min=numpy.append(idx_min[:],idx_max[-1])
        val_min=numpy.append(val_min[:], val_min[-1])
    elif (max(idx_max)< max(idx_min)):
        idx_max=numpy.append(idx_max[:],idx_min[-1])
        val_max=numpy.append(val_max[:], val_max[-1])


    H = numpy.zeros_like(x)
    L = numpy.zeros_like(x)

    #https://localcoder.org/why-does-matlab-interp1-produce-different-results-than-numpy-interp
   # "interp1(x,v,xq) returns interpolated values of a 1-D function at specific query points using linear interpolation.
    # Vector x contains the sample points, and v contains the corresponding values, v(x).
    # Vector xq contains the coordinates of the query points."
    print(idx_max, val_max, t, idx_max.shape, val_max.shape, len(t))
    Max_line=interp1d(idx_max,val_max,kind='linear')(t)
    Min_line=interp1d(idx_min,val_min,kind='linear')(t)
    Lk1=alpha*Max_line[idx_min]+val_min*(1-alpha)
    Lk2=alpha*Min_line[idx_max]+val_max*(1-alpha)


    Lk1=  numpy.vstack(idx_min, Lk1)  #Lk1=[idx_min(:),Lk1(:)];
    #this line means create a matrix(an array in 2d) and stack each element in it's own row
    Lk2=numpy.vstack(idx_max,Lk2)
    Lk=numpy.vstack(Lk1,Lk2)
    #https://www.mathworks.com/matlabcentral/answers/73735-what-does-it-mean-by-writing-idx-in-code
    #~ means discard the first output of the sort function
    #which means it only wants the index
    Lk_col_2 = numpy.argsort(Lk, axis= 0) # sort by first axis?

    Lk_sorted= Lk[Lk_col_2,:] #Lk_sorted=Lk(Lk_col_2,:);


    Lk=Lk_sorted[1:-1,:]

    Lk=numpy.asarray([0,Lk[0,1]],Lk,[x[0].shape,Lk[-1,1]])#Lk=[[1,Lk(1,2)];Lk;[length(x),Lk(end,2)]];


    #%% compute the Lt curve

    idx_Xk=numpy.asarray([1,idx_cb,x[0].shape]) #idx_Xk=[1,idx_cb,length(x)];

    for i in range((idx_Xk[0].shape)-1):
        for j in range (idx_Xk[i],idx_Xk[i+1]):
            kij=(Lk[i+1,1]-Lk[i,1])/(x(idx_Xk[i+1])-x(idx_Xk[i])) #$compute the slope K
            L[j]=Lk[i,1]+kij*(x[j]-x(idx_Xk[i]))



    H[:] = [i for i in x if i not in L]
    print(L,H)
    return L,H



class FilterRun(Thread):
    def __init__(self, rb, pb, channels, processing_size, dtype,work,time,floor,iterations,clean,run):
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





    def write_filtered_data(self):

        numpy.copyto(self.buffer, self.rb.read(self.processing_size).astype(dtype=numpy.float64))
        #self.buffer contains (44100,2) of numpy.float32 audio values sampled with pyaudio
        iterationz = 0

        self.processedrb.write(self.buffer.astype(dtype=self.dtype), error=True)
        Z, freqs, t = mlab.specgram(self.buffer2[:, 0], NFFT=256, Fs=44100, detrend=None, window=None, noverlap=223,
                                        pad_to=None, scale_by_freq=None, mode="default")

        # https://stackoverflow.com/questions/39359693/single-valued-array-to-rgba-array-using-custom-color-map-in-python
        arr_color = self.SM.to_rgba(Z, bytes=False, norm=True)
        arr_color = arr_color[:30, :, :]  # we just want the last bits where the specgram data lies.
        #arr_color = snowy.resize(arr_color, width=60, height=100)  # in the future, this width will be 60.
        arr_color = numpy.rot90(arr_color)  # rotate it and jam it in the buffer lengthwise
        self.cleanspecbuf.growing_write(arr_color)
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

    def __init__(self, sample_rate=44100, channels=2, buffer_delay=1.5,  # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.float32):
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
        self.filterthread = FilterRun(self.rb, self.processedrb, self._channels, self._processing_size, self.dtype,self.work,self.time, self.floor,self.iterations,self.cleanspectrogrambuffer,self.enabled)
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
