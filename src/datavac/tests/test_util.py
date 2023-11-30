import numpy as np
from scipy.stats import linregress

from datavac.util.maths import multiy_singlex_linregress


def test_multiy_singlex_linregress():

    x=np.r_[1,2,3,4]
    ys=np.array([[5,6,7,9],[5,7,9,11]])

    myslopes,myints,myrs=multiy_singlex_linregress(x,ys)
    scslopes,scints,scrs= zip(*[(res.slope,res.intercept,res.rvalue)
                                for res in [linregress(x=x,y=ys[i]) for i in range(ys.shape[0])]])
    assert np.allclose(myslopes,scslopes)
    assert np.allclose(myints,scints)
    assert np.allclose(myrs,scrs)

if __name__=='__main__':
    test_multiy_singlex_linregress()