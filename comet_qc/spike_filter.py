"""
Spike filter.
Replaces: spikefilter_new.pro
"""
import numpy as np


def spike_filter(a: np.ndarray, hwidth: float = 10, std_f: float = 7) -> np.ndarray:
    """
    Remove spikes from a 1-D array using a local median/std-deviation filter.

    A two-pass approach is used:
      Pass 1 (std_f0=1): Create a "clean" working copy by NaN-ing obvious spikes.
      Pass 2 (std_f):    Replace spike values with the local mean from the clean copy.

    NaN values in the input are treated as missing (excluded from statistics).

    Note: Unlike a typical implementation, there is NO std>0 guard on the
    threshold test.  This replicates the IDL behaviour where, when std==0,
    the condition ``abs(a[i] - median) > 0`` is True for any non-equal value,
    so isolated spikes in otherwise-constant data ARE removed.

    Args:
        a:       Input 1-D array.  May contain NaN for pre-existing missing values.
        hwidth:  Half-width (in samples) of the local window.
        std_f:   Standard deviation threshold for spike removal (pass 2).

    Returns:
        Array with spike values replaced by the local mean; same length as *a*.
    """
    a = np.asarray(a, dtype=float)
    na = len(a)
    std_f0 = 1  # threshold for first (rough) pass
    n = 0

    # --- Pass 1: rough identification ---
    abad = a.copy()
    for i in range(na):
        i0 = max(0, i - int(hwidth))
        i1 = min(na - 1, i + int(hwidth))
        if i == 0:
            atmp = a[i + 1: i1 + 1]
        elif i == na - 1:
            atmp = a[i0: i]
        else:
            atmp = np.concatenate([a[i0:i], a[i + 1: i1 + 1]])

        mda = np.nanmedian(atmp)
        std = np.nanstd(atmp)
        if abs(a[i] - mda) > std_f0 * std:
            abad[i] = np.nan
            n += 1

    # --- Pass 2: final replacement ---
    anew = a.copy()
    if n > 0:
        m = 0
        for i in range(na):
            i0 = max(0, i - int(hwidth))
            i1 = min(na - 1, i + int(hwidth))
            if i == 0:
                atmp = abad[i + 1: i1 + 1]
            elif i == na - 1:
                atmp = abad[i0: i]
            else:
                atmp = np.concatenate([abad[i0:i], abad[i + 1: i1 + 1]])

            mna = np.nanmean(atmp)
            mda = np.nanmedian(atmp)
            std = np.nanstd(atmp)
            if abs(a[i] - mda) > std_f * std:
                anew[i] = mna
                m += 1

    return anew
