#A time-causal and time-recursive analogue of the Gabor transform  Lindeberg et al 2024
#implementation by falseywinchnet GPL/MIT 2024 no guarantee if correct
#https://arxiv.org/pdf/2308.14512.pdf
#if lindeberg =  time_causal_stft(data[0:rate],n_fft=512,hop_len=128,tau_max=0.1,c=2,k=4)
#approximal stft = stft(data[0:rate],n_fft=512+64,hop_len=40,window=numpy.hanning(64+512))
import numpy as np

def time_causal_stft(x: np.ndarray, n_fft: int = 512, hop_len: int = 128, tau_max: float = 0.1, c: float = 2.0, K: int = 4) -> np.ndarray:
    # Compute temporal scale levels
    tau = np.array([c**(2 * (k - K)) * tau_max for k in range(1, K + 1)])

    # Compute time constants for recursive filters
    mu = np.sqrt(c**2 - 1) * np.sqrt(tau)
    mu = np.insert(mu, 0, c**(1 - K) * np.sqrt(tau_max))

    # Apply recursive filters to the input signal
    y = x.copy()
    for k in range(K):
        y = apply_recursive_filter(y, mu[k])

    # Compute STFT with adjusted hop size and window size
    hop_len_adj = max(1, int(hop_len * np.sqrt(tau_max)))
    n_fft_adj = max(n_fft, int(n_fft * np.sqrt(tau_max)))
    Zx = stft(y, n_fft=n_fft_adj, hop_len=hop_len_adj, window=np.ones(n_fft_adj))

    # Compute scale-normalized derivatives
    Zx_t = np.sqrt(tau_max) * np.gradient(Zx, axis=1)
    Zx_tt = tau_max * np.gradient(np.gradient(Zx,axis=1), axis=1)

    # Combine scale-normalized derivatives
    Sx = np.abs(Zx) + np.abs(Zx_t) + np.abs(Zx_tt)

    return Sx

def apply_recursive_filter(x: np.ndarray, mu: float) -> np.ndarray:
    y = np.zeros_like(x)
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = y[n - 1] + (x[n] - y[n - 1]) / (1 + mu)
    return y


def stft(x:np.ndarray,n_fft: int, hop_len:int, window: numpy.ndarray) -> np.ndarray:

    xp = np.zeros(x.size+n_fft-1,dtype=np.float64)
    before = n_fft//2
    after = n_fft//2 -1
    xp[before:-after]= x[:]
    xp[0:before] = xp[before+1:(before*2)+1][::-1]
    xp[-after:] = xp[-after*2-1:-before][::-1]

    #Calculate parameters
    n_overlap = n_fft - hop_len
    hop_len = n_fft - n_overlap
    n_segs = (xp.shape[-1] - n_fft) // hop_len + 1
    s20 = int(np.ceil(n_fft / 2))
    s21 = s20 - 1 if (n_fft % 2 == 1) else s20

    # Segmentation
    Sx = np.zeros((n_fft, n_segs), dtype=np.float64)
    strides = (xp.strides[0], hop_len * xp.strides[0])

    starts = np.arange(n_segs) * hop_len

    #dft cisoid centering
    first_half = np.lib.stride_tricks.as_strided(xp[s21:], (s20, n_segs), strides)
    second_half = np.lib.stride_tricks.as_strided(xp, (s21, n_segs), strides)

    Sx[:s20, :] = first_half
    Sx[s20:, :] = second_half
    shift = window.shape[0] // 2 if window.shape[0] % 2 == 0 else (window.shape[0] + 1) // 2
    window = np.concatenate((window[shift:], window[:shift]))
    Sx *= window.reshape(-1, 1) #apply dft-centered windowing

    Zx = numpy.zeros((n_fft//2+1,n_segs),dtype=numpy.complex128)

    for each in range(n_segs):
      Zx[:,each] = numpy.fft.rfft((Sx[:,each]))

    return Zx
