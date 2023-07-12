#translated from the matlab by chatgpt and @Schmitter
import numpy as np
from scipy.signal import savgol_filter


def svmd(signal, maxAlpha=200, tau=0.5, tol=1e-6, stopc=4, init_omega=0):
    # Successive Variational Mode Decomposition
    # authors: Mojtaba Nazari and Sayed Mahmoud Sakhaei
    # mojtaba.nazari.21@gmail.com -- smsakhaei@nit.ac.ir
    # Initial release 2020-5-15 (c) 2020
    #
    #
    #
    #           Input and Parameters:
    #
    # signal     - the 1xN time-series input array (N should be an even number)
    # maxAlpha   - the balancing parameter of the data-fidelity constraint 
    #              (compactness of mode)
    # tau        - time-step of the dual ascent. Set it to 0 in the presence of
    #                high-level noise.
    # tol        - tolerance of convergence criterion; typically around 1e-6
    # stopc      - the type of stopping criteria:
    #                 1- In the Presence of Noise (or recommended for 
    #                    the signals with compact spectrums such as EEG)
    #                 2- For Clean Signal (Exact Reconstruction)
    #                 3- Bayesian Estimation Method
    #                 4- Power of the Last Mode (default)
    # init_omega - initialization type of center frequency (not necessary to 
    #              set):
    #                 0- the center frequencies initiate from 0 (for each mode)
    #                 1- the center frequencies initiate randomly with this
    #                    condition: each new initial value must not be equal to
    #                    the center frequency of previously extracted modes. 
    #  Notice: This method is not sensitive to the center frequency 
    #  initialization and this is considered here just in case (in both cases
    #  the results are usually the same); therefore, it could be ignored.
    #
    #
    #
    #
    #           Output:
    #
    # u       - decomposed modes
    # u_hat   - the spectrum of the decomposed modes
    # omega   - estimated center-frequency of the decomposed modes
    #
    #
    #
    #
    #
    #
    #	Acknowledgments: The SVMD code has been developed by extending the
    #                 variational mode decomposition code that has been made
    #                 public at the following link. 
    #   https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    #                 by K. Dragomiretskiy, D. Zosso.
    #
    #
    #
    #
    # References:
    #[1] M. Nazari, S. M. Sakhaei, "Successive Variational Mode Decomposition,"
    #    Signal Processing, Vol. 174, September 2020.
    #    https://doi.org/10.1016/j.sigpro.2020.107610
    #
    #[2] M. Nazari, S. M. Sakhaei, Variational Mode Extraction: A New Efficient
    #    Method to Derive Respiratory Signals from ECG, IEEE Journal of
    #    Biomedical and Health Informatics, Vol. 22, No. 4, pp. 1059-1067,
    #    july 2018.
    #    http://dx.doi.org/10.1109/JBHI.2017.2734074
    #
    #[3] K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE
    #    Transactions on Signal Processing, vol. 62, pp. 531-544, 2014. 
    #    https://doi.org/10.1109/TSP.2013.2288675

    # ------------ Part 1: Start initializing ------------
    if len(signal) % 2 != 0:
        signal = signal[1:]  # Checking the length of the signal

    y = savgol_filter(signal, 25, 8)  # Filtering the input to estimate the noise
    signoise = signal - y  # Estimating the noise

    save_T = len(signal)
    fs = 1 / save_T

    # Mirroring the signal and noise part to extend
    T = save_T
    f_mir = np.zeros(T * 2)
    f_mir_noise = np.zeros(T * 2)
    f_mir[:T // 2] = signal[T // 2 - 1::-1]
    f_mir_noise[:T // 2] = signoise[T // 2 - 1::-1]
    f_mir[T // 2:3 * T // 2] = signal
    f_mir_noise[T // 2:3 * T // 2] = signoise
    f_mir[3 * T // 2:2 * T] = signal[T:T // 2 - 1:-1]
    f_mir_noise[3 * T // 2:2 * T] = signoise[T:T // 2 - 1:-1]
    f = f_mir
    fnoise = f_mir_noise

    T = len(f)  # Time domain (t -->> 0 to T)
    t = np.arange(1, T + 1) / T

    udiff = tol + np.finfo(float).eps  # Update step
    omega_freqs = t - 0.5 - 1 / T  # Discretization of spectral domain

    # FFT of signal (and Hilbert transform concept = making it one-sided)
    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_onesided = f_hat.copy()
    f_hat_onesided[:T // 2] = 0

    f_hat_n = np.fft.fftshift(np.fft.fft(fnoise))
    f_hat_n_onesided = f_hat_n.copy()
    f_hat_n_onesided[:T // 2] = 0

    noisepe = np.linalg.norm(f_hat_n_onesided, 2) ** 2  # Noise power estimation

    N = 300  # Max. number of iterations to obtain each mode
    omega_L = np.zeros(N)  # Initializing omega_d

    if init_omega == 0:
        omega_L[0] = 0
    else:
        omega_L[0] = np.sort(np.exp(
            np.log(fs) + (
                np.log(0.5) - np.log(fs)
            ) * np.random.rand(1)
        ))

    minAlpha = 10  # The initial value of alpha
    Alpha = minAlpha  # The initial value of alpha
    alpha = np.zeros((1, 1))
    lambda_val = np.zeros((N, len(omega_freqs)), dtype=np.complex128)  # Dual variables vector
    u_hat_L = np.zeros((N, len(omega_freqs)), dtype=np.complex128)  # Keeping changes of mode spectrum
    n = 0  # Main loop counter
    m = 0  # Iteration counter for increasing alpha
    SC2 = 0  # Main stopping criteria index
    l = 0  # The initial number of modes
    bf = 0  # Bit flag to increase alpha
    BIC = np.zeros((1, 1))  # The initial value of Bayesian index
    h_hat_Temp = np.zeros((1, len(omega_freqs)))  # Initialization of filter matrix
    u_hat_Temp = np.zeros((1, len(omega_freqs), 1), dtype=np.complex128)  # Matrix1 of modes
    u_hat_i = np.zeros((1, len(omega_freqs)), dtype=np.complex128)  # Matrix2 of modes
    n2 = 0  # Counter for initializing omega_L
    polm = np.zeros((1, 1))  # Initializing Power of Last Mode index
    omega_d_Temp = np.zeros((1, 1))  # Initialization of center frequencies vector1
    sigerror = np.zeros((1, 1))  # Initializing signal error index for stopping criteria
    gamma = np.zeros((1, 1))  # Initializing gamma
    normind = np.zeros((1, 1))

    # Part 2: Main loop for iterative updates
    while SC2 != 1:
        while Alpha < maxAlpha + 1 and Alpha != np.inf:
            while udiff > tol and n + 1 < N:
                # Update uL
                inter_1 = (Alpha ** 2) * (omega_freqs - omega_L[n]) ** 4
                u_hat_L[n + 1, :] = (
                    f_hat_onesided + inter_1 * u_hat_L[n, :] + lambda_val[n, :] / 2
                ) / (
                    (1 + inter_1) * (
                        (1 + (2 * Alpha) * (
                            omega_freqs - omega_L[n]
                        ) ** 2)
                    ) + np.sum(h_hat_Temp)
                )

                # Update omega_L
                inter_2 = abs(u_hat_L[n + 1, T // 2:T]) ** 2
                omega_L[n + 1] = np.dot(omega_freqs[T // 2:T], inter_2) / np.sum(inter_2)

                # Update lambda (dual ascent)
                lambda_val[n + 1, :] = lambda_val[n, :] + tau * (
                    f_hat_onesided - (
                        u_hat_L[n + 1, :] + (inter_1 * (
                            f_hat_onesided
                            - u_hat_L[n + 1, :]
                            - np.sum(u_hat_i)
                            + lambda_val[n, :]
                            / 2
                        ) - np.sum(u_hat_i)) / (1 + inter_1)
                    ) + np.sum(u_hat_i)
                )

                udiff = np.finfo(float).eps
                udiff = udiff + (1 / T) * np.dot(
                    np.conj(u_hat_L[n + 1, :] - u_hat_L[n, :]),
                    (u_hat_L[n + 1, :] - u_hat_L[n, :])
                ) / (
                    (1 / T) * np.dot(
                        np.conj(u_hat_L[n, :]),
                        u_hat_L[n, :]
                    )
                )

                udiff = abs(udiff)

                n += 1

            # Part 3: Increasing Alpha to achieve a pure mode
            if abs(m - np.log(maxAlpha)) > 1:
                m += 1
            else:
                m += 0.05
                bf = bf + 1
            if bf >= 2:
                Alpha = Alpha + 1
            if Alpha <= (maxAlpha - 1):
                if bf == 1:
                    Alpha = maxAlpha - 1
                else:
                    Alpha = np.exp(m)
                # print(omega_L[n])
                # omega_L = omega_L[n] # init_omega ?

                # Initializing
                udiff = tol + np.finfo(float).eps
                temp_ud = u_hat_L[n, :]  # Keeping the last update of obtained mode
                n = 0  # Loop counter
                lambda_val = np.zeros((N, len(omega_freqs)), dtype=np.complex128)
                u_hat_L = np.zeros((N, len(omega_freqs)), dtype=np.complex128)
                u_hat_L[n, :] = temp_ud

        # Part 4: Saving the Modes and Center Frequencies
        # print(n)
        omega_L[omega_L < 0] = 0
        if l == 0:
            u_hat_Temp[0, :, l] = u_hat_L[n, :]
            omega_d_Temp[l] = omega_L[n - 1]
            alpha[0, l] = Alpha
        else:
            u_hat_Temp = np.append(u_hat_Temp, u_hat_L[n, :].reshape(1,-1, 1), axis=2)
            omega_d_Temp = np.append(omega_d_Temp, omega_L[n - 1])
            alpha = np.append(alpha, [[Alpha]], axis=1)
        Alpha = minAlpha
        bf = 0

        # Initializing omega_L
        if init_omega > 0:
            ii = 0
            while ii < 1 and n2 < 300:
                omega_L = np.sort(np.exp(
                    np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1)
                ))
                checkp = np.abs(omega_d_Temp - omega_L)
                if np.sum(checkp < 0.02) <= 0:
                    ii = 1
                n2 += 1
        else:
            omega_L[:] = 0

        udiff = tol + np.finfo(float).eps  # Update step
        lambda_val = np.zeros((N, len(omega_freqs)), dtype=np.complex128)

        if l == 0:
            gamma[l] = 1
        else:
            gamma = np.append(gamma, 1)

        val = gamma[l] / (
            (alpha[0, l] ** 2) * (omega_freqs - omega_d_Temp[l]) ** 4
        )
        if l == 0:
            h_hat_Temp[l, :] = val
        else:
            h_hat_Temp = np.append(h_hat_Temp, val)

        # Keeping the last desired mode as one of the extracted modes
        val = u_hat_Temp[0, :, l]
        if l == 0:
            u_hat_i[l, :] = val
        else:
            u_hat_i = np.append(u_hat_i, [val], axis=0)

        # Part 5: Stopping Criteria
        if stopc is not None:
            if stopc == 1:
                # In the Presence of Noise
                if np.size(u_hat_i, 0) == 1:
                    sigerror[l] = np.linalg.norm((f_hat_onesided - u_hat_i), 2) ** 2
                else:
                    sigerror = np.append(
                        sigerror,
                        np.linalg.norm((f_hat_onesided - np.sum(u_hat_i, 0)), 2) ** 2
                    )
                if n2 >= 300 or sigerror[l] <= round(noisepe):
                    SC2 = 1
            elif stopc == 2:
                # Exact Reconstruction
                sum_u = np.sum(u_hat_Temp[0, :, :], axis=0)  # Sum of current obtained modes
                val = (1 / T) * np.linalg.norm(sum_u - f_hat_onesided) ** 2 / (
                    (1 / T) * np.linalg.norm(f_hat_onesided) ** 2
                )
                if l == 0:
                    normind[l] = val
                else:
                    normind = np.append(normind, val)
                if n2 >= 300 or normind[l] < 0.005:
                    SC2 = 1
            elif stopc == 3:
                # Bayesian Method
                if np.size(u_hat_i, 0) == 1:
                    sigerror[l] = np.linalg.norm((f_hat_onesided - u_hat_i), 2) ** 2
                else:
                    sigerror = np.append(
                        sigerror,
                        np.linalg.norm((f_hat_onesided - np.sum(u_hat_i, 0)), 2) ** 2
                    )
                val = 2 * T * np.log(sigerror[l]) + (3 * l) * np.log(2 * T)
                if l == 0:
                    BIC[l] = val
                else:
                    BIC = np.append(BIC, val)
                    if BIC[l] > BIC[l - 1]:
                        SC2 = 1
            else:
                # Power of the Last Mode
                val = np.linalg.norm(
                    (
                        4 * Alpha * u_hat_i[l, :]
                        / (1 + 2 * Alpha * (omega_freqs - omega_d_Temp[l]) ** 2)
                    ) * u_hat_i[l, :].conj(),
                    2
                )
                if l == 0:
                    polm[l] = val
                    polm_temp = polm[l]
                    polm[l] = polm[l] / np.max(polm[l])
                else:
                    polm = np.append(polm, val)
                    polm[l] = polm[l] / polm_temp
                    if abs(polm[l] - polm[l - 1]) < tol:
                        SC2 = 1
            # Part 6: Resetting the counters and initializations
            u_hat_L = np.zeros((N, len(omega_freqs)), dtype=np.complex128)
            n = 0  # Reset the loop counter
            l += 1  # (number of obtained modes) + 1
            m = 0
            n2 = 0

    # Part 7: Signal Reconstruction
    omega = omega_d_Temp
    L = len(omega)  # Number of modes
    u_hat = np.zeros((T, L), dtype=np.complex128)
    u_hat[T // 2:T, :] = np.squeeze(u_hat_Temp[0, T // 2:T, :])
    u_hat[T // 2:0:-1, :] = np.squeeze(np.conj(u_hat_Temp[0, T // 2:T, :]))
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros((L, len(t)))
    for l in range(L):
        u[l, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, l])))

    indic = np.argsort(omega)
    omega = omega[indic]
    u = u[indic, :]

    # Remove mirror part
    u = u[:, T // 4:3 * T // 4]

    # Recompute spectrum
    u_hat = np.zeros((save_T, L), dtype=np.complex128)
    for l in range(L):
        u_hat[:, l] = np.conj(np.fft.fftshift(np.fft.fft(u[l, :]))).T

    # Finalize and prepare the output
    return u, u_hat, omega
