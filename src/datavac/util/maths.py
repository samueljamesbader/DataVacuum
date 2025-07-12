from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np
    NDArray2DFloat = np.ndarray[tuple[int, int], np.dtype[np.float64]]
    NDArray1DFloat = np.ndarray[tuple[int], np.dtype[np.float64]]

    NDArray2DInt = np.ndarray[tuple[int, int], np.dtype[np.int_]]
    NDArray1DInt = np.ndarray[tuple[int], np.dtype[np.int_]]
    NDArray2DBool = np.ndarray[tuple[int, int], np.dtype[np.bool_]]

def multiy_singlex_linregress(x,ys):
    """ Run multiple single-variable linear regressions at once against the same x-variable.

    Args:
        x - the single array of predictors (length n) to use for all the regressions
        y - the multiple arrays of outputs (dimension m,n) to regress on

    Returns:
        slopes - an array (length m) of slopes
        intercepts - an array (length m) of intercepts
        rvalues - an array (length m) of rvalues (note: rvalue^2 is coefficient of determination)
    """
    import numpy as np
    from numpy.linalg import lstsq
    ys=ys.T

    # See https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    A=np.vstack([x,np.ones_like(x)]).T
    res,sse,*_=lstsq(A,ys,rcond=None)

    # R-squared formula: https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
    sst=np.sum((ys-np.mean(ys,axis=0))**2,axis=0)
    if len(x)==2:
        r2=1
    else:
        r2=1-sse/sst

    return res[0,:],res[1,:],np.sqrt(r2)

def VTCC(I: np.ndarray, V: np.ndarray, icc: float, itol: float = 1e-14):
    """ Finds the voltage where `I` crosses icc by log-linear interpolation.

    Only the log of (`|I|+itol`) is used, so the sign of I never comes into play.
    The crossing is assumed to go from |`I| < icc` to `|I| >= icc` (off to on).
    If a sweep contains multiple crossings, or crosses in the wrong direction, the
    result should contain a NaN for that sweep.

    This function is vectorized, so `I` and `V` are 2-D arrays of multiple sweeps.

    Args:
        I: an n x m numpy array of currents (n sweeps, m points)
        V: an n x m numpy array of voltages (n sweeps, m points)
        icc: the current defining threshold
        itol: tolerance added to all abs(I) before taking logarithm to prevent log(0)

    Returns:
        an array (of length n) of interpolated crossing voltages
    """
    import numpy as np
    logI=np.log(np.abs(I)+itol)
    logicc=np.log(icc)
    return YatX(X=logI, Y=V, x=logicc)

def YatX(X: NDArray2DFloat, Y: NDArray2DFloat, x: float, reverse_crossing: bool = False) -> NDArray1DFloat:
    """ Finds the `Y` where `X` crosses x by linear interpolation.

    This function is vectorized, so `X` and `Y` are 2-D arrays of multiple sweeps.

    X does not have to be evenly spaced or monotonic, but, for any sweep, if it doesn't cross the point
    exactly once and in the correct direction, the resulting output for that sweep will be NaN.

    Args:
        X: an n x m numpy array of X's (n sweeps, m points)
        Y: an n x m numpy array of Y's (n sweeps, m points)
        x: the target point
        reverse_crossing: if False (default), the crossing is assumed to go from `X` < `x` to `X` > `x`.
            If True, the crossing is assumed to go from `X` > `x` to `X` < `x`.

    Returns:
        an array (of length n) of interpolated targets
    """
    import numpy as np
    X=np.asarray(X); Y=np.asarray(Y);
    if reverse_crossing: X,Y=X[:,::-1],Y[:,::-1]

    ind_aboves=np.nanargmax(X>=x, axis=1)
    ind_belows=X.shape[1]-np.nanargmax(X[:,::-1]<=x, axis=1)-1
    valid_crossing_between=(ind_aboves==(ind_belows+1))
    valid_crossing_rightat=(ind_aboves==ind_belows)

    allinds=np.arange(len(X))
    Ycc=np.empty(len(X))

    # Cover the crossing-between case
    X1,X2=X[allinds,ind_belows][valid_crossing_between],X[allinds,ind_aboves][valid_crossing_between]
    Y1,Y2=Y[allinds,ind_belows][valid_crossing_between],Y[allinds,ind_aboves][valid_crossing_between]

    slope=(Y2-Y1)/(X2-X1)
    Yavg=(Y1+Y2)/2
    Xavg=(X1+X2)/2

    Ycc[valid_crossing_between]=Yavg+(x-Xavg)*slope

    # Cover the crossing-at case
    Ycc[valid_crossing_rightat]=Y[allinds,ind_aboves][valid_crossing_rightat]

    # NaN everything else
    Ycc[~(valid_crossing_between|valid_crossing_rightat)]=np.nan

    return Ycc
