import numpy as np
from scipy.stats import linregress

from datavac.util.maths import multiy_singlex_linregress, YatX


def test_multiy_singlex_linregress():

    x=np.array([1,2,3,4])
    ys=np.array([[5,6,7,9],[5,7,9,11]])

    myslopes,myints,myrs=multiy_singlex_linregress(x,ys)
    scslopes,scints,scrs= zip(*[(res.slope,res.intercept,res.rvalue) # type: ignore
                                for res in [linregress(x=x,y=ys[i]) for i in range(ys.shape[0])]])
    assert np.allclose(myslopes,scslopes)
    assert np.allclose(myints,scints)
    assert np.allclose(myrs,scrs)

def test_yatx():

    # For both X,Y pairs, the x included in X
    X=np.array([[1,2,3,4],[0,2,3,8]])
    Y=np.array([[5,6,7,9],[5,7,9,11]])
    output=YatX(X,Y,x=3)
    assert np.allclose(output,[7,9]), f"output={output}, rather than [7,9]"

    # For both X,Y pairs, the x is between two points
    X=np.array([[1,2,3,5],[0,2,3,6]])
    Y=np.array([[5,6,7,9],[5,7,9,12]])
    output=YatX(X,Y,x=4)
    assert np.allclose(output,[8,10]), f"output={output}, rather than [8,10]"

    # For first X,Y pair, the x included in X
    # for second pair, the x is between two points
    X=np.array([[1,2,3,4],[0,2,4,8]])
    Y=np.array([[5,6,7,9],[5,7,9,11]])
    output=YatX(X,Y,x=3)
    assert np.allclose(output,[7,8]), f"output={output}, rather than [7,8]"

    # Endpoint usage
    # For first X,Y pair, the x=X[0]
    # for second pair, the x=X[-1]
    X=np.array([[1,2,3,4],[-4,-3,0,1]])
    Y=np.array([[5,6,7,9],[5,7,9,11]])
    output=YatX(X,Y,x=1)
    assert np.allclose(output,[5,11]), f"output={output}, rather than [5,11]"

    # For first X,Y pair, the x included in X
    # for second pair, the x is out-of-range
    X=np.array([[1,2,3,4],[4,5,6,7]])
    Y=np.array([[5,6,7,9],[5,7,9,11]])
    output=YatX(X,Y,x=3)
    assert np.isclose(output[0],7) and np.isnan(output[1]), f"output={output}, rather than [7,NaN]"


    # Same as previous but reversed
    X=np.array([[4,3,2,1],[7,6,5,4]])
    Y=np.array([[9,7,6,5],[11,9,7,5]])
    output=YatX(X,Y,x=3,reverse_crossing=True)
    assert np.isclose(output[0],7) and np.isnan(output[1]), f"output={output}, rather than [7,NaN]"

if __name__=='__main__':
    #test_multiy_singlex_linregress()
    test_yatx()
