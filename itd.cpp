/*
Copyright 2024 joshuah rainstar <joshuah.rainstar@gmail.com>

Permission is hereby granted to any person obtaining 
a copy of this software and associated documentation files (the “Software”),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

Money, women, or prayers to god soliciting money and women for the author must be provided at some point!

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

  Intrinsic Time-Scale Decomposition (ITD). ITD is a patented decomposition method developed under grant.
  NIH/NINDS grants nos. 1R01NS046602-01 and 1R43NS39240-01. 
  This algorithm is patented US-7457756-B1 by Frei And Osorio, 2027-02-16 Adjusted expiration

  This adaptation of ITD has been extremely simplified to run fast.
  it may not be numerically accurate or suitable for your uses. 
  Two versions are offered, one of which finds a common baseline in two different sets of data,
  and the other of which finds a baseline in 1d data.

  it is recommended that for the use with continuous data, the processing method is as follows:
  use a circular buffer with modulous tracking to rotate the samples.
  re-assess extrema in the entire buffer every iteration
  use from the last extrema in the first buffer to the first extrema in the last buffer
  set the first and last baseline knots manually to said values
  update the j array
  compute only the baseline[i] array for the inner third of the buffer overall
  rotate buffers, rinse and repeat
  

  The retention and reuse of extrema to reprocess data before and after adjustments, or along multiple channels,
  is provided, such that disagreements on splines will not effect processing. 
  simply estimate the extrema the first time you process your data with this method, and retain the extrema.
  further iterations should reuse the extrema but evaluate and produce the baseline on new data.
*/

static int extrema[FFTBins];
static float baseline_knots[FFTBins];
static int j_lookup[FFTBins];
static float b[FFTBins];
static float d[FFTBins];
static float u[FFTBins];
static float v[FFTBins];
static float h[FFTBins];
static int extrema[FFTBins];
static float baseline[FFTBins];

void itd_baseline_extract_iq(complex_t* data, float* baseline,int length,int* index, bool compute_extrema) {
	int& idx = *index;  // Create a reference to *idx

	// Find extrema for I and Q
	if (compute_extrema == true) {
		idx = 0;
    memset(baseline, 0, sizeof(baseline));  // Set all elements to 0 whenever we start over
    memset(extrema, 0, sizeof(extrema));
    memset(baseline_knots, 0, sizeof(baseline_knots));
    memset(j_lookup, 0, sizeof(j_lookup));
    memset(b, 0, sizeof(b));
    memset(d, 0, sizeof(d));
    memset(u, 0, sizeof(u));
    memset(v, 0, sizeof(v));
    memset(h, 0, sizeof(h));
    
		for (int i = 1; i < length - 1; i++) {
			if ((((data[i - 1].re < data[i].re) && (data[i].re >= data[i + 1].re)) ||
				((data[i - 1].re > data[i].re) && (data[i].re <= data[i + 1].re))) &&
				(((data[i - 1].im < data[i].im) && (data[i].im >= data[i + 1].im)) ||
					((data[i - 1].im > data[i].im) && (data[i].im <= data[i + 1].im)))) {
				extrema[idx] = i;
				idx++;
			}
		}
	}
	if (idx < 2) {
		return; // break early
	}
	float alpha = 0.5f;

	for (int k = 1; k < idx - 1; k++) {
		int prev_idx = extrema[k - 1];
		int curr_idx = extrema[k];
		int next_idx = extrema[k + 1];

		// I and Q values at the extrema
		float I_prev = data[prev_idx].re, I_curr = data[curr_idx].re, I_next = data[next_idx].re;
		float Q_prev = data[prev_idx].im, Q_curr = data[curr_idx].im, Q_next = data[next_idx].im;

		// Average I and Q at each extremum to get a common scalar baseline
		float avg_prev = (I_prev + Q_prev) / 2.0f;
		float avg_curr = (I_curr + Q_curr) / 2.0f;
		float avg_next = (I_next + Q_next) / 2.0f;

		// Time indices of extrema
		float t_prev = (float)prev_idx, t_curr = (float)curr_idx, t_next = (float)next_idx;

		// Weighting factor based on distance between extrema
		float weight = (t_curr - t_prev) / (t_next - t_prev);

		// Calculate the common scalar baseline at the current extremum
		baseline_knots[k] = alpha * (avg_prev + weight * (avg_next - avg_prev)) + (1.0f - alpha) * avg_curr;
	}

	// Set the first and last baseline knots
	baseline_knots[0] = (data[extrema[0]].re + data[extrema[0]].im) / 2.0f;
	baseline_knots[idx] = (data[extrema[idx]].re + data[extrema[idx]].im) / 2.0f;

	for (int i = 0; i < idx; i++) {
		h[i] = (float)(extrema[i + 1] - extrema[i]);
	}

	for (int i = 1; i < idx; i++) {
		u[i] = h[i - 1] / (h[i - 1] + h[i]);
		v[i] = 1.0f - u[i];
		b[i] = 6.0f * ((baseline_knots[i + 1] - baseline_knots[i]) / h[i] - (baseline_knots[i] - baseline_knots[i - 1]) / h[i - 1]) / (h[i - 1] + h[i]);
	}

	for (int i = 1; i < idx; i++) {
		d[i] = 2.0f;
		b[i] = b[i] - u[i] * b[i - 1];
		d[i] = d[i] - u[i] * v[i - 1];
		u[i] = u[i] / d[i];
		b[i] = b[i] / d[i];
	}

	for (int i = idx - 2; i >= 0; i--) {
		b[i] = b[i] - v[i] * b[i + 1];
	}

	int j = 0;
	for (int i = 0; i < length; i++) {
		while (j < idx - 1 && extrema[j + 1] <= i) {
			j++;
		}
		j_lookup[i] = j;
	}

	for (int i = 0; i < length; i++) {
		j = j_lookup[i];
		float t = (float)(i - extrema[j]) / h[j];
		baseline[i] = (1 - t) * baseline_knots[j] + t * baseline_knots[j + 1] +
			h[j] * h[j] / 6.0f * ((1 - t) * (1 - t) * (1 - t) - 1 + t) * b[j] +
			h[j] * h[j] / 6.0f * (t * t * t - t) * b[j + 1];
	}
}

void itd_baseline_extract(float* data, float* baseline,int length,int* index, bool compute_extrema) {
	int& idx = *index;  // Create a reference to *idx

	// Find extrema
	if (compute_extrema == true) {
		idx = 0;
		for (int i = 1; i < length - 1; i++) {
			if ((((data[i - 1] < data[i]) && (data[i] >= data[i + 1])) ||
				((data[i - 1] > data[i]) && (data[i] <= data[i + 1]))) {
				extrema[idx] = i;
				idx++;
			}
		}
	}
	if (idx < 2) {
		return; // break early
	}
	float alpha = 0.5f;

	for (int k = 1; k < idx - 1; k++) {
		int prev_idx = extrema[k - 1];
		int curr_idx = extrema[k];
		int next_idx = extrema[k + 1];

		// I and Q values at the extrema
		float I_prev = data[prev_idx], I_curr = data[curr_idx], I_next = data[next_idx];

		// Average I and Q at each extremum to get a common scalar baseline
		float avg_prev = I_prev;
		float avg_curr = I_curr;
		float avg_next = I_next;

		// Time indices of extrema
		float t_prev = (float)prev_idx, t_curr = (float)curr_idx, t_next = (float)next_idx;

		// Weighting factor based on distance between extrema
		float weight = (t_curr - t_prev) / (t_next - t_prev);

		// Calculate the common scalar baseline at the current extremum
		baseline_knots[k] = alpha * (avg_prev + weight * (avg_next - avg_prev)) + (1.0f - alpha) * avg_curr;
	}

	// Set the first and last baseline knots
	baseline_knots[0] = data[extrema[0]];
	baseline_knots[idx] = data[extrema[idx]];

	for (int i = 0; i < idx; i++) {
		h[i] = (float)(extrema[i + 1] - extrema[i]);
	}

	for (int i = 1; i < idx; i++) {
		u[i] = h[i - 1] / (h[i - 1] + h[i]);
		v[i] = 1.0f - u[i];
		b[i] = 6.0f * ((baseline_knots[i + 1] - baseline_knots[i]) / h[i] - (baseline_knots[i] - baseline_knots[i - 1]) / h[i - 1]) / (h[i - 1] + h[i]);
	}

	for (int i = 1; i < idx; i++) {
		d[i] = 2.0f;
		b[i] = b[i] - u[i] * b[i - 1];
		d[i] = d[i] - u[i] * v[i - 1];
		u[i] = u[i] / d[i];
		b[i] = b[i] / d[i];
	}

	for (int i = idx - 2; i >= 0; i--) {
		b[i] = b[i] - v[i] * b[i + 1];
	}

	int j = 0;
	for (int i = 0; i < length; i++) {
		while (j < idx - 1 && extrema[j + 1] <= i) {
			j++;
		}
		j_lookup[i] = j;
	}

	for (int i = 0; i < length; i++) {
		j = j_lookup[i];
		float t = (float)(i - extrema[j]) / h[j];
		baseline[i] = (1 - t) * baseline_knots[j] + t * baseline_knots[j + 1] +
			h[j] * h[j] / 6.0f * ((1 - t) * (1 - t) * (1 - t) - 1 + t) * b[j] +
			h[j] * h[j] / 6.0f * (t * t * t - t) * b[j + 1];
	}
}
