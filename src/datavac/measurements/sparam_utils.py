import numpy as np


def s2y(s11,s12,s21,s22, z0=50):
    y0=1/z0
    deltas=(1+s11)*(1+s22)-s12*s21
    y11=((1-s11)*(1+s22)+s12*s21)/deltas * y0
    y12=-2*s12/deltas * y0
    y21=-2*s21/deltas * y0
    y22=((1+s11)*(1-s22)+s12*s21)/deltas * y0
    return y11,y12,y21,y22


def y2s(y11,y12,y21,y22, z0=50):
    delta=(1+z0*y11)*(1+z0*y22)-z0**2*y12*y21
    s11=((1-z0*y11)*(1+z0*y22)+z0**2*y12*y21)/delta
    s12=-2*z0*y12/delta
    s21=-2*z0*y21/delta
    s22=((1+z0*y11)*(1-z0*y22)+z0**2*y12*y21)/delta
    return s11,s12,s21,s22


def s2z(s11,s12,s21,s22, z0=50):
    delta=(1-s11)*(1-s22)-s12*s21
    z11=((1+s11)*(1-s22)+s12*s21)/delta *z0
    z12=2*s12/delta * z0
    z21=2*s21/delta * z0
    z22=((1-s11)*(1+s22)+s12*s21)/delta *z0
    return z11, z12, z21, z22


def z2s(z11,z12,z21,z22, z0=50):
    delta=(z11+z0)*(z22+z0)-z12*z21
    s11=((z11-z0)*(z22+z0)-z12*z21)/delta
    s12=2*z0*z12/delta
    s21=2*z0*z21/delta
    s22=((z11+z0)*(z22-z0)-z12*z21)/delta
    return s11, s12, s21, s22


def a2y(a11,a12,a21,a22):
    deltaa=a11*a22-a12*a21
    y11=a22/a12
    y12=-deltaa/a12
    y21=-1/a12
    y22=a11/a12
    return y11, y12, y21, y22


def y2a(y11,y12,y21,y22):
    deltay=y11*y22-y12*y21
    a11=-y22/y21
    a12=-1/y21
    a21=-deltay/y21
    a22=-y11/y21
    return a11, a12, a21, a22


def z2a(z11,z12,z21,z22):
    deltaz=z11*z22-z12*z21
    a11=z11/z21
    a12=deltaz/z21
    a21=1/z21
    a22=z22/z21
    return a11, a12, a21, a22


def _backwards_conjunctive_transform(c11,c12,c21,c22, t11,t12,t21,t22):
    # C' = T C T+ (Hillbrand & Russer 1976)
    # https://doi.org/10.1109/TCS.1976.1084200

    m11=t11*c11+t12*c21; m12=t11*c12+t12*c22
    m21=t21*c11+t22*c21; m22=t21*c12+t22*c22
    r11=m11*np.conj(t11)+m12*np.conj(t12)
    r12=m11*np.conj(t21)+m12*np.conj(t22)
    r21=m21*np.conj(t11)+m22*np.conj(t12)
    r22=m21*np.conj(t21)+m22*np.conj(t22)
    return r11, r12, r21, r22


def cy2ca(cy11,cy12,cy21,cy22, a12,a22):
    # resulting chain from original admittance: T=[[0,a12],[1,a22]]
    return _backwards_conjunctive_transform(cy11,cy12,cy21,cy22, 0,a12,1,a22)


def ca2cy(ca11,ca12,ca21,ca22, y11,y21):
    # resulting admittance from original chain: T=[[-y11,1],[-y21,0]]
    return _backwards_conjunctive_transform(ca11,ca12,ca21,ca22, -y11,1,-y21,0)


def cy2cz(cy11,cy12,cy21,cy22, z11,z12,z21,z22):
    # resulting impedance from original admittance: T=[[z11,z12],[z21,z22]]
    return _backwards_conjunctive_transform(cy11,cy12,cy21,cy22, z11,z12,z21,z22)


def cz2cy(cz11,cz12,cz21,cz22, y11,y12,y21,y22):
    # resulting admittance from original impedance: T=[[y11,y12],[y21,y22]]
    return _backwards_conjunctive_transform(cz11,cz12,cz21,cz22, y11,y12,y21,y22)

def cz2ca(cz11,cz12,cz21,cz22, a11,a21):
    # resulting chain from original impedance: T=[[1,-a11],[0,-a21]]
    return _backwards_conjunctive_transform(cz11,cz12,cz21,cz22, 1,-a11,0,-a21)