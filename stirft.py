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
