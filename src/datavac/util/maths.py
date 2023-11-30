import numpy as np
from numpy.linalg import lstsq


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


def VTCC(I, V, icc, itol=1e-14):
    logI=np.log(np.abs(I)+itol)
    logicc=np.log(icc)

    ind_aboves=np.argmax(logI>logicc, axis=1)
    ind_belows=logI.shape[1]-np.argmax(logI[:,::-1]<logicc, axis=1)-1
    valid_crossing=(ind_aboves==(ind_belows+1))

    allinds=np.arange(len(I))
    lI1,lI2=logI[allinds,ind_belows],logI[allinds,ind_aboves]
    V1,V2=V[allinds,ind_belows],V[allinds,ind_aboves]

    slope=(lI2-lI1)/(V2-V1)
    Vavg=(V1+V2)/2
    lIavg=(lI1+lI2)/2

    VTcc=np.empty_like(Vavg)
    VTcc[valid_crossing]=Vavg[valid_crossing]+(logicc-lIavg[valid_crossing])/slope[valid_crossing]
    VTcc[~valid_crossing]=np.NaN

    return VTcc
