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

def sparam_helper(df):
    df['freq'] = np.real(df['freq'])
    if 's11' in df and 'S11' not in df:
        df.rename(columns={'s11':'S11','s12':'S12','s21':'S21','s22':'S22'},inplace=True)
    if 'S11Mag' in df and 'S11' not in df:
        for w in ['11','12','21','22']:
            df[f'S{w}'] = df[f'S{w}Mag'] * np.exp(1j*df[f'S{w}Angle']) # assume angle is in radians

    # print(df[['S11', 'S12', 'S21', 'S22']])
    if 'Y11' not in df:
        df['Y11'], df['Y12'], df['Y21'], df['Y22'] = s2y(df['S11'], df['S12'], df['S21'], df['S22'])
    if 'Z11' not in df:
        df['Z11'], df['Z12'], df['Z21'], df['Z22'] = s2z(df['S11'], df['S12'], df['S21'], df['S22'])


def simple_rf_mosfet_extraction(df,width):

    sparam_helper(df)

    # h21 is a current ratio, so 20x log
    df[f'|h21| [dB]']=20*np.log10(np.abs(df['Y21']/df['Y11']))
    df['fT_extr [GHz]']=df['freq']*np.power(10,df['|h21| [dB]']/20)/1e9

    # https://en.wikipedia.org/wiki/Mason%27s_invariant#Derivation_of_U
    # U is already a power ratio so just 10x log
    re=np.real; im=np.imag
    with np.errstate(invalid='ignore'):
        df[f'U [dB]']=10*np.log10(
            (np.abs(df['Y21']-df['Y12'])**2 /
             (4*(re(df['Y11'])*re(df['Y22'])-re(df['Y12'])*re(df['Y21'])))))
    df['fMax_extr [GHz]']=df['freq']*np.power(10,df['U [dB]']/20)/1e9

    # https://www.microwaves101.com/encyclopedias/stability-factor
    Delta=df['S11']*df['S22']-df['S12']*df['S21']
    K = (1-np.abs(df['S11'])**2-np.abs(df['S22'])**2+np.abs(Delta)**2)/(2*np.abs(df['S21']*df['S12']))

    # this formula with 1/(K+sqrt(K^2-1)) is less common but more robust for large K
    # according to Microwaves 101 and easy to show it's equal.
    k2m1=np.clip(K**2-1,0,np.inf) # we only use the K>1 values of MAG anyway, so clip to avoid sqrt(-)
    MAG = (1/(K+np.sqrt(k2m1))) * np.abs(df['S21'])/np.abs(df['S12'])
    MSG = np.abs(df['S21'])/np.abs(df['S12'])
    df['K']=K
    df['MAG [dB]']=10*np.log10(np.choose(MAG>0,[np.nan,MAG]))
    df['MSG [dB]']=10*np.log10(MSG)
    df['MAG-MSG [dB]']=10*np.log10(np.choose(K>=1,[MSG,MAG]))

    # RF small-signal circuit parameters
    Wum=width*1e6

    fF=1e-15; uS=1e-6
    w=2*np.pi*df['freq']
    #df['tanThetaY12']=im(df['Y12']) / re(df['Y12'])
    #df['tanThetaY11']=im(df['Y11']) / re(df['Y11'])
    #df['tanThetaY22']=im(df['Y22']) / re(df['Y22'])
    df['Cgg/W [fF/um]']= im(df['Y11']) / w / Wum /fF
    df['Cgd/W [fF/um]']=-im(df['Y12']) / w / Wum /fF
    df['Cgs/W [fF/um]']= im(df['Y11'] + df['Y12']) / w / Wum /fF
    df['Cds/W [fF/um]']= im(df['Y22'] + df['Y12']) / w / Wum /fF
    df['Rds*W [Ohm.um]']=1/re(df['Y22']+df['Y12']) * Wum
    df['GM/W [uS/um]']=np.abs(df['Y21']-df['Y12']) / Wum / uS
    Rs=df['Rs [Ohm.um]']=re(df['Z12']) * Wum
    df['Rd*W [Ohm.um]']=(re(df['Z22'])-Rs) * Wum
    df['Rg*W [Ohm.um]']=(re(df['Z11'])-Rs) * Wum
    df['GM/2πCgs [GHz]']=df['GM/W [uS/um]']/(2*np.pi*df['Cgs/W [fF/um]']) #uS/fF=GHz


def deembed(subdfd_raw, open_dfd, short_dfd):
    # For the moment this is done by caller
    #sparam_helper(subdfd_raw)
    #sparam_helper(open_dfd)
    #sparam_helper(short_dfd)

    # Using Open-Short deembedding
    # https://scikit-rf.readthedocs.io/en/latest/tutorials/Deembedding.html
    try:
        assert np.allclose(short_dfd['freq'], open_dfd['freq']), "Frequencies of short and open deembedders do not match"
        assert np.allclose(subdfd_raw['freq'], open_dfd['freq']), "Frequencies of subdevice and open deembedders do not match"
    except Exception as e:
        raise Exception("Frequency arrays of deembedders and subdevice do not match") from e

    y11_s, y12_s, y21_s, y22_s = short_dfd['Y11'], short_dfd['Y12'], short_dfd['Y21'], short_dfd['Y22']
    y11_o, y12_o, y21_o, y22_o = open_dfd['Y11'], open_dfd['Y12'], open_dfd['Y21'], open_dfd['Y22']
    y11_s, y12_s, y21_s, y22_s = (y11_s - y11_o), (y12_s - y12_o), (y21_s - y21_o), (y22_s - y22_o)
    z11_s, z12_s, z21_s, z22_s = s2z(*y2s(y11_s, y12_s, y21_s, y22_s))
    
    y11_m, y12_m, y21_m, y22_m = subdfd_raw['Y11'], subdfd_raw['Y12'], subdfd_raw['Y21'], subdfd_raw['Y22']
    y11_m, y12_m, y21_m, y22_m = (y11_m - y11_o), (y12_m - y12_o), (y21_m - y21_o), (y22_m - y22_o)
    z11_m, z12_m, z21_m, z22_m = s2z(*y2s(y11_m, y12_m, y21_m, y22_m))
    z11_m, z12_m, z21_m, z22_m = (z11_m - z11_s), (z12_m - z12_s), (z21_m - z21_s), (z22_m - z22_s)

    deembeded=dict(zip(['freq','S11','S12','S21','S22'], [subdfd_raw['freq'],*z2s(z11_m, z12_m, z21_m, z22_m)]))
    sparam_helper(deembeded)
    return deembeded