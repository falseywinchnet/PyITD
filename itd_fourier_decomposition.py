"""copyright 2024 joshuah rainstar  joshuah.rainstar@gmail.com
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import numba

@numba.jit(nopython=True)
def generate_sine_wave(freq: float, sample_rate: float, duration: float) -> np.ndarray:
    t = np.arange(0, duration, 1/sample_rate)
    return np.sin(2 * np.pi * freq * t)

@numba.jit(nopython=True)
def find_extrema(signal: np.ndarray) -> np.ndarray:
    extrema = np.zeros(signal.size, dtype=np.int64)
    idx = 0
    extrema[idx] = 0  # Always start with an extrema at zero
    idx += 1

    for i in range(1, signal.size-1):
        if (signal[i] > 0 and 0 > signal[i+1]) or \
           (signal[i] < 0 and 0 < signal[i+1]):
            extrema[idx] = i
            idx += 1

    extrema[idx] = extrema[idx-1]*2 - extrema[idx-2]
    idx += 1
    return extrema,idx

def itd_sine_wrapper(signal: np.ndarray, sample_rate: int):
    problem = signal.copy()
    duration = len(signal) / sample_rate
    frequencies = np.arange(2,sample_rate//2-1, 96)[::-1]
    products = []
    for freq in range(1,frequencies.size):
            sine_wave = generate_sine_wave(frequencies[freq], sample_rate, duration)
            extrema ,idx = find_extrema(sine_wave)
            baseline = itd_baseline_extract_fast(problem, extrema,idx)
            rotation = problem - baseline
            products.append(rotation)
            problem = problem - rotation
    products.append(problem)
    return products

@numba.jit()
def itd_baseline_extract_fast(I: np.ndarray, extrema_input:np.ndarray, idx:int):
    n = I.size
    baseline_knots = np.zeros(n,  dtype=np.float64)
    baseline = np.zeros(n,  dtype=np.float64)
    j_lookup = np.zeros(n, dtype=np.int64)
    b = np.zeros(n)
    d = np.zeros(n)
    u = np.zeros(n)
    v = np.zeros(n)
    h = np.zeros(n)

    alpha = 0.5

    for k in range(1, idx - 1):
        prev_idx = extrema_input[k - 1]
        curr_idx = extrema_input[k]
        next_idx = extrema_input[k + 1]

        # I and Q values at the extrema
        # Average I and Q at each extremum to get a common scalar baseline

        avg_prev = I[prev_idx]
        avg_curr = I[curr_idx]

        avg_next = I[next_idx]
        # Time indices of extrema
        t_prev, t_curr, t_next = prev_idx, curr_idx, next_idx

        # Weighting factor based on distance between extrema
        weight = (t_curr - t_prev) / (t_next - t_prev)

        # Calculate the common scalar baseline at the current extremum
        baseline_knots[k] = alpha * (avg_prev + weight * (avg_next - avg_prev)) + (1 - alpha) * avg_curr

    # Set the first and last baseline knots
    baseline_knots[0], baseline_knots[idx] = I[extrema_input[0]], I[extrema_input[idx]]

    for i in range(idx):
        h[i] = extrema_input[i+1] - extrema_input[i]

    for i in range(1, idx):
        u[i] = h[i-1] / (h[i-1] + h[i])
        v[i] = 1 - u[i]
        b[i] = 6 * ((baseline_knots[i+1] - baseline_knots[i]) / h[i] - (baseline_knots[i] - baseline_knots[i-1]) / h[i-1]) / (h[i-1] + h[i])

    for i in range(1, idx):
        d[i] = 2
        b[i] = b[i] - u[i] * b[i-1]
        d[i] = d[i] - u[i] * v[i-1]
        u[i] = u[i] / d[i]
        b[i] = b[i] / d[i]

    for i in range(idx-2, -1, -1):
        b[i] = b[i] - v[i] * b[i+1]

        # At the start of the spline calculation
    b[0] = 0  # Natural spline condition at start
    b[idx-1] = 0  # Natural spline condition at end
    #remove second order derivatives at endpoints

    j = 0
    for i in range(n):
        while j < idx - 1 and extrema_input[j+1] <= i:
            j += 1
        j_lookup[i] = j

    for i in range(n):
      j = j_lookup[i]
      t = (i - extrema_input[j]) / h[j]
      if j == idx-2:  # Last segment
          baseline[i] = (1-t)*baseline_knots[j] + t*baseline_knots[j+1]  # Linear only
      else:
          baseline[i] = (1-t)*baseline_knots[j] + t*baseline_knots[j+1] + h[j]*h[j]/6 * ((1-t)**3-1+t)*b[j] + h[j]*h[j]/6 * (t**3-t)*b[j+1]
    return baseline

#this modified form of the intrinsic time-scale decomposition algorithm is not patented
#use itd_sine_wrapper to decompose your signal into a set of frequency-governed bands
#i would appreciate research collaboration and feedback on the effectiveness of this method for decomposition
#please apply it to your research problems in mechanical engineering and vibrational-data analysis
#this can be nondeterministic due to some floating point operations caused by very small numbers
#but the final product usually perfectly sums back up to the input

def fourier_mode_decomposition_valid(rotation):
    x = np.fft.fft(rotation)
    a = np.abs(x)
    half_len = len(a) // 2
    
    # Find all peaks in first half (excluding endpoints)
    peaks = []
    for i in range(1, half_len-1):
        if a[i] > a[i-1] and a[i] > a[i+1]:
            peaks.append((i, a[i]))
    
    if len(peaks) < 3:  # Need at least 3 peaks
        return np.zeros(rotation.size)
        
    # Sort peaks by amplitude and get indices
    peak_indices = [i for i, _ in sorted(peaks, key=lambda x: x[1], reverse=True)]
    peak_max = peak_indices[0]
    
    # Find valid peaks before and after maximum
    valid_before = [i for i in peak_indices if i < peak_max - 1]
    valid_after = [i for i in peak_indices if i > peak_max + 1]
    
    if not valid_before or not valid_after:
        return np.zeros(rotation.size)
        
    # Use closest valid peaks
    first_peak = max(valid_before)
    last_peak = min(valid_after)
    
    # Original mode extraction logic
    mina = first_peak + np.argmin(a[first_peak:peak_max+1])
    minb = peak_max + np.argmin(a[peak_max:last_peak+1])
    
    xn = np.zeros(len(a), dtype=np.complex64)
    xn[mina:minb] = x[mina:minb]
    xn[-minb:-mina] = x[-minb:-mina]
    
    return np.fft.ifft(xn).real
    

def fourier_mode_decomposition_any(rotation):
  #note: ths approach does not attempt to find valid peaks.
  #it only attempts to cleanly isolate fourier components.
  #a corrected approach results in a lot more possible decompositions,
  #because there is simply much more data to examine.

    x = numpy.fft.fft(rotation)
    a = numpy.abs(x)
    half_len = len(a) // 2
    xn = np.zeros(len(a), dtype=np.complex64)

    # Get the highest peak index in 'a' which is not the first element and within the first half of 'a'
    peak_max = np.argmax(a[1:half_len]) + 1

    # Check if we have a valid peak_max
    if peak_max == 1 or peak_max == half_len - 1:
        return numpy.zeros(rotation.size)  # Return zeros if conditions aren't met

    # Find the highest peak index before it
    first_peak = np.argmax(a[:peak_max])

    # Find the highest peak index between peak_max and the midpoint of 'a'
    last_peak = np.argmax(a[peak_max+1:half_len]) + peak_max + 1

    # Check if we have distinct peaks
    if first_peak == peak_max-1 or last_peak == peak_max+1:
        return numpy.zeros(rotation.size)  # Return zeros if conditions aren't met

    # Find the min index between first_peak and peak_max
    mina = first_peak + np.argmin(a[first_peak:peak_max+1])

    # Find the min index between peak_max and last_peak
    minb = peak_max + np.argmin(a[peak_max:last_peak+1])

    # Transpose by slicing the range of [mina:minb] and [-minb:-mina] to 'x'
    xn[mina:minb] = x[mina:minb]
    xn[-minb:-mina] = x[-minb:-mina]

    return numpy.fft.ifft(xn).real


def itd_fourier_decomposition(signal: np.ndarray, sample_rate: int):
   """
   Full cascade decomposition with iterative ITD and Fourier extraction.
   """
   fourier_modes = []  
   source_indices = [] 
   final_output = []   
   
   has_modes = True
   current_signal = signal.copy()
   iteration = 1
   
   while has_modes:
       has_modes = False
       modes_this_iter = 0
       rotations = itd_sine_wrapper(current_signal, sample_rate)
       
       # Extract Fourier modes from each rotation except the residual
       for idx, rotation in enumerate(rotations[:-1]):  
           mode = fourier_mode_decomposition_any(rotation)
           if not np.allclose(mode, 0):
               has_modes = True
               modes_this_iter += 1
               fourier_modes.append(mode)
               source_indices.append(idx)
               rotations[idx] = rotation - mode
       
       if has_modes:
           print(f"Iteration {iteration}: Found {modes_this_iter} Fourier modes")
           current_signal = np.sum(rotations, axis=0)
           iteration += 1
       else:
           print("No more Fourier modes found, finalizing decomposition...")
           # Build final output array
           for i in range(len(rotations)-1):  
               for mode_idx, source_idx in enumerate(source_indices):
                   if source_idx == i:
                       final_output.append(fourier_modes[mode_idx])
               final_output.append(rotations[i])
           
           final_output.append(rotations[-1])
   
   print(f"Total decomposition complete: {len(fourier_modes)} Fourier modes extracted over {iteration-1} iterations")
   return final_output


def itd_fourier_decomposition_lean(signal: np.ndarray, sample_rate: int):
    """
    Full cascade decomposition with iterative ITD and Fourier extraction.
    Maintains one Fourier mode array per rotation by accumulating modes.
    Returns alternating [mode, rotation] pairs followed by residual.
    """
    current_signal = signal.copy()
    iteration = 1
    num_modes = 0  # Keep track of total modes for final print
    
    # Initialize modes array to match number of rotations (excluding residual)
    rotations = itd_fourier_wrapper(current_signal, sample_rate)
    accumulated_modes = np.zeros((len(rotations)-1,) + signal.shape)
    
    has_modes = True
    while has_modes:
        has_modes = False
        modes_this_iter = 0
        rotations = itd_fourier_wrapper(current_signal, sample_rate)
        
        # Extract Fourier modes from each rotation except the residual
        for idx, rotation in enumerate(rotations[:-1]):  
            mode = fourier_mode_decomposition_any(rotation)
            if not np.allclose(mode, 0):
                has_modes = True
                modes_this_iter += 1
                num_modes += 1
                accumulated_modes[idx] += mode
                rotations[idx] = rotation - mode
        
        if has_modes:
            print(f"Iteration {iteration}: Found {modes_this_iter} Fourier modes")
            current_signal = np.sum(rotations, axis=0)
            iteration += 1
        else:
            print("No more Fourier modes found, finalizing decomposition...")
            # Build final output array with alternating [mode, rotation] pairs
            final_output = []
            for i in range(len(rotations)-1):
                final_output.append(accumulated_modes[i])
                final_output.append(rotations[i])
            
            final_output.append(rotations[-1])  # Add residual
    
    print(f"Total decomposition complete: {num_modes} Fourier modes extracted over {iteration-1} iterations")
    return final_output

#this method will iteratively find all components in a signal
#their validity is not guaranteed due to non-deterministic behavior of float operations on very small numbers
#itd_fourier_decomposition can use a lot more memory but produces all elements

#itd_fourier_decomposition_lean uses less memory but only uses/holds one mode array per rotation.
#final output structure is : [modes1,rotation1,modesn,rotationn...finalresidual]
#note: the number of iterations can be large 

#fourier_mode_decomposition_valid produces much more valid isolations but on large signals will never really converge

#overall, the behavior here will be: 
#residual rotational data is odd, fourier products are even
#fourier products will resemble a narrowbanded windowing with very sharp dropoff, sometimes
#residual products will not, and may show some imaging/aliasing
#additionally, due to the fourier method's reliance on peak availability, the natural ramification is that
#decomposition will be slightly dc dominated resulting in difficulty finding peaks for lower(later) rotations.
#as a result the fourier modes tend to be high frequncy dominant and the residual tends to be low frequency dominant.
#alternating successive decompositions that start high, then low, may alleviate this problem.

#TODO: improve the determinism of this process. At this time, the number of products varies from iteration to iteration.
#getting this to behave more consistently will provide better utilization of this as an alternative to wavelets.
#also, wavelets- applying wavelets to this could be interesting

