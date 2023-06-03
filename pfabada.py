"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possibles moothed  models  of
the  data introduced,  obtaining  an  estimation  of the  underlying
signal that is statistically  compatible  with the  noisy  measurements.
based on P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"
Copyright (C) 2007 Free Software Foundation, Inc. 
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
"""

"""PFABADA(parametric Fabada) is a somewhat optimized noise reduction technique based on FABADA
that uses known properties of signals and skimage's sigma estimator to converge on a best
fit approximation in the presence of unknown variance and normalization parameters,
generalizing the fabada approach for most data streams, and using numba for acceleration.
Copywrite 2022 Joshuah Rainstar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
#version idk 11/14/2022 this fixes any divide by zero issues


"""
The approach the authors use for the stopping criterion is based on chi-square statistics, where they calculate a chi-square statistic (chi2_data) and the associated PDF. This approach may be perfectly fine in theory, but it's clear that it's causing numerical stability issues in practice.

Given that we're already in a Bayesian setting, one option might be to replace this with a more Bayesian-style criterion. For instance, we could consider stopping the iterations when the posterior distribution has stabilized, as measured by some appropriate distance metric (Kullback-Leibler divergence might be a reasonable choice here).

This kind of approach would go something like this:

Calculate the posterior distribution after each iteration.
Measure the Kullback-Leibler (KL) divergence between the posterior distributions from successive iterations.
Stop iterating when the KL divergence drops below some threshold, indicating that the posterior distribution has effectively stabilized.
The KL divergence between two Gaussian distributions with means µ1, µ2 and standard deviations σ1, σ2 is given by:

python
Copy code
def kl_divergence(mu1, mu2, sigma1, sigma2):
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5
We could then check the KL divergence after each iteration:

kl_divergence = kl_divergence(mu_current, mu_previous, sigma_current, sigma_previous)
if kl_divergence < threshold:
    break  # Stop iterating

Implementing this in your code might look something like this:

python
Copy code
while 1:
    # ... existing code ...

    # after calculating new posterior_mean and posterior_variance, 
    # calculate the KL divergence from the previous iteration
    if iteration > 0:  # don't calculate KL divergence on the first iteration
        kl_divergence = kl_divergence(mu_previous, posterior_mean, sigma_previous, posterior_variance)
        if kl_divergence < threshold:  # threshold is a hyperparameter you'll need to set
            break

    # store current posterior_mean and posterior_variance for next iteration
    mu_previous = posterior_mean.copy()
    sigma_previous = posterior_variance.copy()

    # ... remaining code ...
Yes, you could initialize mu_previous and sigma_previous to arrays of ones, 
or you could initialize them as copies of your initial posterior_mean and posterior_variance, prior to entering the loop. Either approach could work.


""
import numpy
import numba
import scipy
from pywt import dwtn

@numba.jit(numba.float64[:](numba.float64[:]),cache=True,nogil=True)
def numba_fabada(data: numpy.ndarray) -> (numpy.ndarray):
    x = numpy.zeros_like(data,dtype=numpy.float64)
    x[:] = data.copy()
    x[numpy.where(numpy.isnan(data))] = 0
    iterations: int = 1
    N = x.size
    max_iterations = 1000

    bayesian_weight = numpy.zeros_like(x,dtype=numpy.float64)
    bayesian_model = numpy.zeros_like(x,dtype=numpy.float64)
    model_weight = numpy.zeros_like(x,dtype=numpy.float64)

    # pre-declaring all arrays allows their memory to be allocated in advance
    posterior_mean = numpy.zeros_like(x,dtype=numpy.float64)
    posterior_mean[:] = x.copy()

    initial_evidence = numpy.zeros_like(x,dtype=numpy.float64)
    evidence = numpy.zeros_like(x,dtype=numpy.float64)
    prior_mean = numpy.zeros_like(x,dtype=numpy.float64)
    prior_variance = numpy.zeros_like(x,dtype=numpy.float64)
    posterior_variance = numpy.zeros_like(x,dtype=numpy.float64)

    chi2_data_min = N
    data_variance = numpy.zeros_like(x,dtype=numpy.float64)
    with numba.objmode(sigma=numba.float64):  # annotate return type
        coeffs = dwtn(x, wavelet='db2')
        detail_coeffs = coeffs['d' * x.ndim]
        sigma = numpy.median(numpy.abs(detail_coeffs)) / 0.6616518484657332 #scipy.stats.gamma.ppf(0.75,0.5)
        # https://github.com/scikit-image/scikit-image/blob/main/skimage/restoration/_denoise.py#L938-L1008
        #note: 2d and higher dimension sigmas will require calling the function instead of inlining wavelets

    data_variance.fill(sigma**2)
    data_variance[numpy.where(numpy.isnan(data))] = 1e-15
    data_variance[data_variance==0] = 1e-15

    posterior_variance[:] = data_variance.copy()
    prior_variance[:] = data_variance.copy()

    prior_mean[:] = x.copy()

    # fabada figure 14
    #formula 3, but for initial assumptions

    upper = numpy.square(numpy.sqrt(data_variance)*-1)
    lower = 2 * data_variance
    first = (-upper / lower)
    second = numpy.sqrt(2 * numpy.pi) * data_variance
    evidence[:] = numpy.exp(first) / second
    initial_evidence[:] = evidence.copy()

    evidence_previous = numpy.mean(evidence)

    while 1:
        
        # GENERATES PRIORS
        for i in numba.prange(N - 1):
            prior_mean[i] = (posterior_mean[i - 1] + posterior_mean[i] + posterior_mean[i + 1]) / 3

        prior_mean[0] = (posterior_mean[0] + (posterior_mean[1] + posterior_mean[2]) / 2) / 3
        prior_mean[-1] = (posterior_mean[-1] + (posterior_mean[-2] + posterior_mean[-3]) / 2) / 3

        prior_variance = posterior_variance.copy() #if this is an array, you must use .copy() or it will
        #cause any changes made to posterior_variance to also automatically be applied to prior
        #variance, making these variables redundant.

        # APPLY BAYES' THEOREM
        # fabada figure 8?
        for i in numba.prange(N):
            if prior_variance[i] > 0:
           # posterior_variance[i] = 1 / (1 / data_variance[i] + 1 / prior_variance[i])
                posterior_variance[i] = (data_variance[i] * prior_variance[i])/(data_variance[i] + prior_variance[i])
            else:
                posterior_variance[i] = 0
            #saves on instructions- replaces three divisions, 1 add with one mult, 1 div, 1 add

        # fabada figure 7
        for i in numba.prange(N):
            if prior_variance[i] > 0 and posterior_variance[i] > 0:
                posterior_mean[i] = (
                ((prior_mean[i] / prior_variance[i]) + (x[i] / data_variance[i])) * posterior_variance[i])
            else:
                posterior_mean[i] = prior_mean[i] #the variance cannot be reduced further

        upper = numpy.square(prior_mean - x)
        lower = 2 * (prior_variance + data_variance)
        first =((-upper/lower))
        second = numpy.sqrt(2*numpy.pi) * prior_variance + data_variance

        evidence = numpy.exp(first) / second

        # fabada figure 6: probability distribution calculation

        evidence_derivative = numpy.mean(evidence) - evidence_previous
        evidence_previous = numpy.mean(evidence)

        # EVALUATE CHI2
        chi2_data = numpy.sum((x - posterior_mean) ** 2 / data_variance)


        if iterations == 1:
            chi2_data_min = chi2_data

        # COMBINE MODELS FOR THE ESTIMATION

        for i in numba.prange(N):
            model_weight[i] = evidence[i] * chi2_data

        for i in numba.prange(N):
            bayesian_weight[i] = bayesian_weight[i] + model_weight[i]
            bayesian_model[i] = bayesian_model[i] + (model_weight[i] * posterior_mean[i])


        if ((chi2_data > N) and (evidence_derivative < 0)) \
                or (iterations > max_iterations):  # don't overfit the data
            break
        iterations = iterations + 1
        # COMBINE ITERATION ZERO
    for i in numba.prange(N):
        model_weight[i] = initial_evidence[i] * chi2_data_min
    for i in numba.prange(N):
        bayesian_weight[i] = bayesian_weight[i]+ model_weight[i]
        bayesian_model[i] = bayesian_model[i] + (model_weight[i] * x[i])


    for i in numba.prange(N):
        if bayesian_weight[i] > 0:
            x[i] = bayesian_model[i] / bayesian_weight[i]
        else:
          x[i] = x[i]


    return x


#here is the logic needed for a 2d nearest neighbors smoothing algo.
#this, along with replacing loop-specific logic in 1d with 2d, allows fabada to be generalized to the 2d domain.
        normal = posterior_mean.copy()
        transposed = posterior_mean.copy()
        transposed = transposed.T
        transposed_raveled = numpy.ravel(transposed)
        normal_raveled = numpy.ravel(normal)

        target_a = numpy.zeros_like(normal_raveled)
        target_b = numpy.zeros_like(normal_raveled)

        # GENERATES PRIORS
        for i in range(elements - 1):
            target_a[:] = (normal_raveled[i - 1] + normal_raveled[i] + normal_raveled[i + 1]) / 3

        target_a[0] = (normal_raveled[0] + (normal_raveled[1] + normal_raveled[2]) / 2) / 3
        target_a[-1] = (normal_raveled[-1] + (normal_raveled[-2] + normal_raveled[-3]) / 2) / 3
        normal_raveled[:] = target_a[:]

        for i in range(elements - 1):
            target_b[:] = (transposed_raveled[i - 1] + transposed_raveled[i] + transposed_raveled[i + 1]) / 3

        target_b[0] = (transposed_raveled[0] + (transposed_raveled[1] + transposed_raveled[2]) / 2) / 3
        target_b[-1] = (transposed_raveled[-1] + (transposed_raveled[-2] + transposed_raveled[-3]) / 2) / 3
        transposed_raveled[:] = target_b[:]
        transposed = transposed.T
        prior_mean = (transposed + normal)/2 
