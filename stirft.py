def compute_synthesis_window(analysis_window, hop):
    """
    Computes the optimal synthesis window given an analysis window
    and hop (frame shift). The procedure is described in

    D. Griffin and J. Lim, *Signal estimation from modified short-time Fourier transform,*
    IEEE Trans. Acoustics, Speech, and Signal Process.,
    vol. 32, no. 2, pp. 236-243, 1984.

    Parameters
    ----------
    analysis_window: array_like
        The analysis window
    hop: int
        The frame shift
    """

    norm = np.zeros_like(analysis_window)
    L = analysis_window.shape[0]

    # move the window back as far as possible while still overlapping
    n = 0
    while n - hop > -L:
        n -= hop

    # now move the window and sum all the contributions
    while n < L:
        if n == 0:
            norm += analysis_window**2
        elif n < 0:
            norm[: n + L] += analysis_window[-n - L :] ** 2
        else:
            norm[n:] += analysis_window[:-n] ** 2

        n += hop

    return analysis_window / norm



import numpy as np
def stirft(x:np.ndarray, window:np.ndarray) -> np.ndarray:
    
    n_fft = 512 #also the segment length
    win_length = 512
    hop_len = 128    
      

    xp = np.zeros(x.size+4*hop_len-1,dtype=np.float64)
    xp[(hop_len*2):-(hop_len*2-1)]= x[:]
    xp[0:(hop_len*2)] = xp[(hop_len*2+1):((hop_len*2)*2)+1][::-1]
    xp[-(hop_len*2-1):] = xp[-(hop_len*2-1)*2-1:-(hop_len*2)][::-1]
    
    #Calculate parameters
    n_overlap = n_fft - hop_len
    n_segs = (xp.shape[-1] - n_fft) // hop_len + 1
    s20 = int(np.ceil(n_fft / 2))
    s21 = s20 - 1 if (n_fft % 2 == 1) else s20

    # Segmentation
    Sx = np.zeros((n_fft, n_segs), dtype=np.float64)
    strides = (xp.strides[0], hop_len * xp.strides[0])

    starts = np.arange(n_segs) * hop_len
    first_half = np.lib.stride_tricks.as_strided(xp, (s21, n_segs), strides)
    second_half = np.lib.stride_tricks.as_strided(xp[s21:], (s20, n_segs), strides)

    Sx[:s20, :] = first_half
    Sx[s20:, :] = second_half
    #note: for dft centering on stft, we transpose the first and second half
    #additionally, we perform fftshift on the window
    #however, for the stirft we do not perform this



    Sx *= window.reshape(-1, 1) #apply windowing
    Zx = np.zeros((n_fft,n_segs),dtype=np.float64)

    Zx[:] = np.fft.irfft(Sx, axis=0)[:n_fft]

    return Zx
def istirft(Sx: np.ndarray,persistent_buffer:np.ndarray,window:np.ndarray):


    #persistent buffer is 384 samples here
    n_fft = 512
    win_len = 512
    hop_len = 128
    N = Sx.shape[1]*hop_len

    Xout = np.zeros((n_fft*2-2,Sx.shape[1]),dtype=Sx.dtype)
    Xout[:n_fft,:] = Sx[:]
    Xout[n_fft:,:]=numpy.flip(Sx[1:n_fft-1,:],axis=0)

    xbuf = np.fft.rfft(Sx, n=n_fft*2-2, axis=0).real
    #perform fftshift here along first axis if dft centered stft

    x = np.zeros(N, dtype=np.float64) 

    n = 0
    for i in range(xbuf.shape[1]):
        processing = xbuf[:, i] * window
        out = processing[0 : hop_len]  # fresh output samples
        out[:128] += persistent_buffer[:128]
        persistent_buffer[: -hop_len] = persistent_buffer[hop_len:]  # shift out left
        persistent_buffer[-hop_len :] = 0.0
        persistent_buffer[:] += processing[-384:] #n_FFT - hop_length
        x[n : n + hop_len] = out[:]
        n += hop_len
    return x, persistent_buffer


msewin= compute_synthesis_window(np.hanning(512),128) 
s = stirft(x=data[0:rate],window=msewin)#note the use of the MSE window for the forward transform, this is intentional
i,buf = istirft(s,numpy.zeros(384),np.hanning(512)*2) #note the window, this is intentional
plt.plot(data[128:rate-384]) #note that the first segment will not overlap right because of tapering oscillation
plt.plot(i[384:])
plt.plot(data[128:rate-384][0:512])
plt.plot(i[384:][0:512])


import torch
import torch

def compute_synthesis_window_torch(analysis_window, hop):
    norm = torch.zeros_like(analysis_window)
    L = analysis_window.shape[0]

    n = 0
    while n - hop > -L:
        n -= hop

    while n < L:
        if n == 0:
            norm += analysis_window**2
        elif n < 0:
            norm[: n + L] += analysis_window[-n - L :] ** 2
        else:
            norm[n:] += analysis_window[:-n] ** 2

        n += hop

    return analysis_window / norm


def stirft(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
  """
  Short-time Inverse Fourier Transform (STFT) implementation with Torch.

  Args:
    x: Input signal as a 1D torch.Tensor.
    window: Window function as a 1D torch.Tensor.

  Returns:
    Short-time Fourier Transform of the signal as a 2D torch.Tensor.
  """

  n_fft = 512
  win_length = 512
  hop_len = 128

  # Padding and mirroring
  xp = torch.zeros(x.size(0) + 4*hop_len - 1, dtype=torch.float64)
  xp[(hop_len*2):-(hop_len*2-1)] = x[:]
  xp[0:(hop_len * 2)] = torch.flip(xp[(hop_len * 2 + 1):((hop_len * 2) * 2 + 1)], dims=[0])

  xp[-(hop_len * 2 - 1):] = torch.flip(xp[-((hop_len * 2 - 1) * 2 + 1):-(hop_len * 2)], dims=[0])

  # Calculate parameters
  n_overlap = n_fft - hop_len
  n_segs = (xp.shape[-1] - n_fft) // hop_len + 1
  s20 = int(np.ceil(n_fft / 2))
  s21 = s20 - 1 if (n_fft % 2 == 1) else s20

  # Segmentation
  Sx = torch.zeros((n_fft, n_segs), dtype=torch.float64)

    # Manual segmentation similar to using stride tricks
  for i in range(n_segs):
      start = i * hop_len
      # For the first half, take segments from the start
      if start + s21 <= xp.size(0):
        Sx[:s21, i] = xp[start:start + s21]
      # For the second half, adjust the start index because xp[s21:] was used in NumPy
      start += s21
      if start + s20 <= xp.size(0):
        Sx[s21:, i] = xp[start:start + s20]

  # Windowing
  Sx *= window.reshape(-1, 1)

  # Inverse FFT
  Zx = torch.fft.irfft(Sx, dim=0)[:n_fft]

  return Zx



def istirft(Sx: torch.Tensor, persistent_buffer: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
  """
  Inverse Short-time Fourier Transform (ISTFT) implementation with Torch.

  Args:
    Sx: Spectrogram as a 2D torch.Tensor.
    persistent_buffer: Internal buffer as a 1D torch.Tensor.
    window: Window function as a 1D torch.Tensor.

  Returns:
    Reconstructed signal as a 1D torch.Tensor.
  """

  n_fft = 512
  win_len = 512
  hop_len = 128
  N = Sx.shape[1] * hop_len

  # Prepare output
  Xout = torch.zeros((n_fft*2-2,Sx.shape[1]), dtype=Sx.dtype)
  Xout[:n_fft,:] = Sx[:]
  Xout[n_fft:,:]=torch.flip(Sx[1:n_fft-1,:],dims=[0])

  # Real-valued FFT and shift
  xbuf = torch.fft.rfft(Sx, n=n_fft*2-2, dim=0).real

  # Initialize output signal
  x = torch.zeros(N, dtype=torch.float64)

  n = 0
  for i in range(xbuf.shape[1]):
        processing = xbuf[:, i] * window
        out = processing[0:hop_len]
        out[:128] += persistent_buffer[:128]
        persistent_buffer_clone = persistent_buffer.clone()
        persistent_buffer[:-hop_len] = persistent_buffer_clone[hop_len:]  
        persistent_buffer[-hop_len:] = 0.0
        persistent_buffer += processing[-384:]
        x[n:n + hop_len] = out
        n += hop_len

  return x,persistent_buffer

msewin= compute_synthesis_window_torch(torch.hann_window(512),128)
input = torch.from_numpy(data[0:rate]).float()
s = stirft(x=input,window=msewin)
t,buf = istirft(s,torch.zeros(384),torch.hann_window(512)*2)
plt.plot(data[128:rate-384])
plt.plot(t.numpy()[384:])
