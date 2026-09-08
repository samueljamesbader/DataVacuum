from datavac.measurements.sparam_utils import ca2cy, cy2cz, cz2cy, s2y, s2z, y2s, z2s
import numpy as np

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


def nparam_helper(df):
    sparam_helper(df)

    from scipy.constants import Boltzmann as kb
    Ts=290 # to match IEEE definition, source temperature is 290K

    # TODO: these add headers to the DF, gotta use the right functions for that!!!
    if 'CY11' not in df:
        ca11 = 4*kb*Ts*df['Rn']
        if 'ReGammaOpt' in df and 'ImGammaOpt' in df:
            GammaOpt=df['ReGammaOpt']+1j*df['ImGammaOpt']
        elif 'GammaOptMag' in df and 'GammaOptAngle' in df:
            GammaOpt=df['GammaOptMag']*np.exp(1j*df['GammaOptAngle'])
        else:
            raise ValueError("Cannot compute CY without either ReGammaOpt/ImGammaOpt or GammaOptMag/GammaOptAngle")
        YOpt=1/50*(1-GammaOpt)/(1+GammaOpt)
        Fmin=10**(df['NFmin']/10)
        ca12 = 4*kb*Ts*((Fmin-1)/2-df['Rn']*np.conj(YOpt))
        ca21 = 4*kb*Ts*((Fmin-1)/2-df['Rn']*YOpt)
        ca22 = 4*kb*Ts*(df['Rn']*np.abs(YOpt)**2)
        cy11, cy12, cy21, cy22 = ca2cy(ca11, ca12, ca21, ca22, y11=df['Y11'], y21=df['Y21'])
        df['CY11'], df['CY12'], df['CY21'], df['CY22'] = cy11, cy12, cy21, cy22
        df['CY11Mag'], df['CY12Mag'], df['CY21Mag'], df['CY22Mag'] = np.abs(cy11), np.abs(cy12), np.abs(cy21), np.abs(cy22)
        df['CY11Angle'], df['CY12Angle'], df['CY21Angle'], df['CY22Angle'] = np.angle(cy11), np.angle(cy12), np.angle(cy21), np.angle(cy22)

    if 'NFmin' not in df:
        Rn=np.real(1/(4*kb*Ts)*df['CY22']/np.abs(df['Y21'])**2)

        Yc=-df['Y21']*df['CY12']/df['CY22']+df['Y11']
        Gc=np.real(Yc)
        Bc=np.imag(Yc)

        Gu=np.real(df['CY11']-np.abs(df['CY12'])**2/df['CY22'])/(4*kb*Ts)

        Gopt=np.sqrt(Gc**2+Gu/Rn)
        Bopt=-Bc

        Fmin=1+2*Rn*(Gopt+Gc)

        #df['Fmin']=Fmin
        df['NFmin'] = 10*np.log10(Fmin)
        df['Gopt']=Gopt
        df['Bopt']=Bopt
        df['Rn']=Rn
        Z0=50
        YOptNorm=Z0*(Gopt+1j*Bopt)
        GammaOpt=(1-YOptNorm)/(1+YOptNorm)
        df['ReGammaOpt']=np.real(GammaOpt)
        df['ImGammaOpt']=np.imag(GammaOpt)

    if not ('ReGammaOpt' in df and 'ImGammaOpt' in df):
        GammaOpt=df['GammaOptMag']*np.exp(1j*df['GammaOptAngle'])
        df['ReGammaOpt']=np.real(GammaOpt)
        df['ImGammaOpt']=np.imag(GammaOpt)

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

    # Deembed the short itself from the open
    y11_s, y12_s, y21_s, y22_s = (y11_s - y11_o), (y12_s - y12_o), (y21_s - y21_o), (y22_s - y22_o)
    z11_s, z12_s, z21_s, z22_s = s2z(*y2s(y11_s, y12_s, y21_s, y22_s))

    # Deembed the device from the open
    y11_m, y12_m, y21_m, y22_m = subdfd_raw['Y11'], subdfd_raw['Y12'], subdfd_raw['Y21'], subdfd_raw['Y22']
    y11_m, y12_m, y21_m, y22_m = (y11_m - y11_o), (y12_m - y12_o), (y21_m - y21_o), (y22_m - y22_o)

    # Further deembed the device from the short
    z11_m, z12_m, z21_m, z22_m = s2z(*y2s(y11_m, y12_m, y21_m, y22_m))
    z11_m, z12_m, z21_m, z22_m = (z11_m - z11_s), (z12_m - z12_s), (z21_m - z21_s), (z22_m - z22_s)

    deembeded=dict(zip(['freq','S11','S12','S21','S22'], [subdfd_raw['freq'],*z2s(z11_m, z12_m, z21_m, z22_m)]))
    sparam_helper(deembeded)
    return deembeded

def deembed_noise(subdfd_raw, open_dfd, short_dfd):
    # For the moment this is done by caller
    #sparam_helper(subdfd_raw)
    #sparam_helper(open_dfd)
    #sparam_helper(short_dfd)
    #nparam_helper(subdfd_raw)

    
    # K. Aufinger and J. Bock, "A Straightforward Noise De-Embedding Method and its Application to High-Speed Silicon Bipolar Transistors,"
    # ESSDERC '96: Proceedings of the 26th European Solid State Device Research Conference, Bologna, Italy, 1996, pp. 957-960.

    from scipy.constants import Boltzmann as kb
    T=300 # assumes measurement is at room temperature
    
    y11_s, y12_s, y21_s, y22_s = short_dfd['Y11'], short_dfd['Y12'], short_dfd['Y21'], short_dfd['Y22']
    y11_o, y12_o, y21_o, y22_o = open_dfd['Y11'], open_dfd['Y12'], open_dfd['Y21'], open_dfd['Y22']

    # Bosma's Theorem, but enforcing reciprocity for the Y-matrices (avoids complex noise parameters!)
    cy11_s, cy12_s, cy21_s, cy22_s = 4*kb*T*short_dfd['Y11'].real, 2*kb*T*(short_dfd['Y12']+short_dfd['Y21']).real,\
                                     2*kb*T*(short_dfd['Y21']+short_dfd['Y12']).real, 4*kb*T*short_dfd['Y22'].real
    cy11_o, cy12_o, cy21_o, cy22_o = 4*kb*T* open_dfd['Y11'].real, 2*kb*T*( open_dfd['Y12']+ open_dfd['Y21']).real,\
                                     2*kb*T*( open_dfd['Y21']+ open_dfd['Y12']).real, 4*kb*T* open_dfd['Y22'].real

    # Deembed the short itself from the open, including the noise correlation matrix
    y11_s, y12_s, y21_s, y22_s = (y11_s - y11_o), (y12_s - y12_o), (y21_s - y21_o), (y22_s - y22_o)
    z11_s, z12_s, z21_s, z22_s = s2z(*y2s(y11_s, y12_s, y21_s, y22_s))
    cy11_s, cy12_s, cy21_s, cy22_s = (cy11_s - cy11_o), (cy12_s - cy12_o), (cy21_s - cy21_o), (cy22_s - cy22_o)
    cz11_s, cz12_s, cz21_s, cz22_s = cy2cz(cy11_s, cy12_s, cy21_s, cy22_s, z11=z11_s, z12=z12_s, z21=z21_s, z22=z22_s)

    # Deembed the device from the open, including the noise correlation matrix
    y11_m, y12_m, y21_m, y22_m = subdfd_raw['Y11'], subdfd_raw['Y12'], subdfd_raw['Y21'], subdfd_raw['Y22']
    y11_m, y12_m, y21_m, y22_m = (y11_m - y11_o), (y12_m - y12_o), (y21_m - y21_o), (y22_m - y22_o)
    cy11_m, cy12_m, cy21_m, cy22_m = subdfd_raw['CY11'], subdfd_raw['CY12'], subdfd_raw['CY21'], subdfd_raw['CY22']
    cy11_m, cy12_m, cy21_m, cy22_m = (cy11_m - cy11_o), (cy12_m - cy12_o), (cy21_m - cy21_o), (cy22_m - cy22_o)

    # Further deembed the device from the short, including the noise correlation matrix
    z11_m, z12_m, z21_m, z22_m = s2z(*y2s(y11_m, y12_m, y21_m, y22_m))
    cz11_m, cz12_m, cz21_m, cz22_m = cy2cz(cy11_m, cy12_m, cy21_m, cy22_m, z11=z11_m, z12=z12_m, z21=z21_m, z22=z22_m)
    z11_m, z12_m, z21_m, z22_m = (z11_m - z11_s), (z12_m - z12_s), (z21_m - z21_s), (z22_m - z22_s)
    cz11_m, cz12_m, cz21_m, cz22_m = (cz11_m - cz11_s), (cz12_m - cz12_s), (cz21_m - cz21_s), (cz22_m - cz22_s)


    deembeded=dict(zip(['freq','S11','S12','S21','S22'], [subdfd_raw['freq'],*z2s(z11_m, z12_m, z21_m, z22_m)]))|\
              dict(zip(['CY11','CY12','CY21','CY22'], [subdfd_raw['freq'],*cz2cy(cz11_m, cz12_m, cz21_m, cz22_m, y11=y11_m, y12=y12_m, y21=y21_m, y22=y22_m)]))
    sparam_helper(deembeded)
    nparam_helper(deembeded)
    return deembeded