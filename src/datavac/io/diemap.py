import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import pickle

from datavac.logging import logger
from datavac.util import check_dtypes


def read_velox_wafermap(file,mask,read_sites=True):
    with open(file) as file:
        dies=[]
        sites=[]
        for l in file:
            if l.startswith("XIndex="):
                xindex=float(l.split("=")[1].strip())/1e3
            if l.startswith("YIndex="):
                yindex=float(l.split("=")[1].strip())/1e3
            if l.startswith("Diameter="):
                diameter=float(l.split("=")[1].strip())
            if l.startswith("Origin="):
                origin=l.split("=")[1].strip()
            if l.startswith("FlatAngle="):
                flatangle=int(l.split("=")[1].strip())
            if l.startswith("WaferTestAngle="):
                wafertestangle=int(l.split("=")[1].strip())
            if l.startswith("[Die]"):
                for l in file:
                    if l.startswith("["):
                        break
                    else:
                        if l.split(",")[-1].strip()!="X":
                            dies.append([int(c) for c in l.split("=")[1].split(",")[:2]])
            if read_sites:
                if l.startswith("[SubDie]"):
                    for l in file:
                        if l.startswith("["):
                            break
                        else:
                            assert int(l.split("=")[0])==len(sites)
                            try:
                                sites.append(dict(zip(['X','Y','rowname','rowrep','DUT'],
                                                      l.split("=")[1].split(",")[:2] + \
                                                      l.split(",")[-1].split("-")[:2] + \
                                                      [int(l.split("DUT")[-1].strip())])))
                            except:
                                if len(sites)!=0:
                                    print(f"Site {len(sites)} could not be parsed from wafermap")
                                sites.append(None)

    assert ((origin,wafertestangle) in [('LL',0),('UR',270),('UL',0)]), \
        f"Never ran this combination of Origin {origin}, WaferTestAngle {wafertestangle} before.  " \
        "I'm unsure of the die coordinate mapping."
    assert flatangle in [90,270], \
        f"Never ran this flatangle {flatangle} before. 90 is NotchRight, 270 is NotchLeft."


    if (origin,wafertestangle)==('LL',0):
        dietable=pd.DataFrame({'DieX':[die[0] for die in dies],'DieY':[die[1] for die in dies]})
        centerdiex=np.median(dietable['DieX'])
        centerdiey=np.median(dietable['DieY'])
        x=dietable['DieX']-centerdiex
        y=dietable['DieY']-centerdiey
    if (origin,wafertestangle)==('UL',0): # ... not validated
        logger.warning(f"Origin {origin}, WaferTestAngle {wafertestangle} has not been validated")
        dietable=pd.DataFrame({'DieX':[die[0] for die in dies],'DieY':[die[1] for die in dies]})
        centerdiex=np.median(dietable['DieX'])
        centerdiey=np.median(dietable['DieY'])
        x=dietable['DieX']-centerdiex
        y=-(dietable['DieY']-centerdiey)
    if (origin,wafertestangle)==('UR',270):
        #logger.debug('Flipped XY because of origin/testangle')
        # Because of wafertestangle I think, Velox flips the order in which X and Y appear in the die list
        dietable=pd.DataFrame({'DieX':[die[1] for die in dies],'DieY':[die[0] for die in dies]})
        centerdiex=np.median(dietable['DieX'])
        centerdiey=np.median(dietable['DieY'])
        # And, because of origin and wafer testangle, -Y is up and -X is right
        x=-(dietable['DieY']-centerdiey)
        y=-(dietable['DieX']-centerdiex)
    # If notch right, flip both signs
    if flatangle==270:
        #print('Flipped XY sign because this is notch right')
        x=-x
        y=-y

    assert np.isclose(np.round(centerdiex-.5),centerdiex-.5), f"Not sure how to find centerdiex {centerdiex}"
    assert np.isclose(np.round(centerdiey-.5),centerdiey-.5), f"Not sure how to find centerdiey {centerdiey}"

    dietable['DieLoc']=pd.Series([f"x{dlx}y{dly}" for dlx,dly in zip(dietable['DieX'],dietable['DieY'])],dtype="string")
    dietable['DieRadius [mm]']=np.round(np.sqrt((x*xindex)**2+(y*yindex)**2),decimals=1)
    dietable['DieCenterX [mm]']=x*xindex
    dietable['DieCenterY [mm]']=y*yindex
    x-=.5
    y-=.5
    x=np.asarray(np.round(x),dtype='int')
    y=np.asarray(np.round(y),dtype='int')
    dietable['DieLB']=pd.Series([f"L{xi:+d}B{yi:+d}" for xi,yi in zip(x,y)],dtype='string')

    check_dtypes(dietable)
    return {'sites':sites,'dietable':dietable,'mask':mask,'name':file.name}

# TODO: the names in this class are confusing and non-standard, and the caching is done at instance level as remnant
# from when it was PROJECT-based
class DieMapBook:
    DIEMAP_DIR=Path(os.environ['DATAVACUUM_DIEMAP_DIR'])
    def __init__(self):
        self._die_locators={}
        self._dielbs={}

    def get_diemap(self,mask,map):
        if (mask,map) not in self._die_locators:
            self._die_locators[(mask,map)]=read_velox_wafermap(
                self.DIEMAP_DIR/map,
                read_sites=False, mask=mask)
        return self._die_locators[(mask,map)]

    def list_diemaps(self,mask):
        return [f.name for f in self.DIEMAP_DIR.glob("*.map")]

    def get_dielb(self,mask):
        if mask not in self._dielbs:
            with open(self.DIEMAP_DIR/f"{mask}_Diemap-info.pkl",'rb') as f:
                self._dielbs[mask]=pickle.load(f)
        return self._dielbs[mask]
