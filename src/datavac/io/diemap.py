import os
from pathlib import Path

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#from matplotlib import patches

import pickle

from datavac.io.database import get_database
from datavac.util.logging import logger
from datavac.util.tables import check_dtypes
from datavac.io.make_diemap import make_fullwafer_diemap as standalone_mfd


def make_fullwafer_diemap(name, xindex, yindex, radius=150, notchsize=5, plot=False, save=True):
    standalone_mfd(name,xindex=xindex,yindex=yindex,
                   radius=radius,notchsize=notchsize,plot=plot,save=save,
                   labeller=(lambda x,y:f"L{x:+d}B{y:+d}"))

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
            if l.startswith("RefDieOffset="):
                refdiex,refdiey=[int(xory.strip()) for xory in l.split("=")[1].split(',')]
            if l.startswith("DieRefPos="):
                # See below where centerdiex, centerdiey are assigned
                if l.split("=")[1].strip().endswith("Top"):
                    y_off_to_center_ll=.5
                elif l.split("=")[1].strip().endswith("Bottom"):
                    y_off_to_center_ll=-.5
                else:
                    raise Exception(f"What is {l}?")
                if l.split("=")[1].strip().startswith("Left"):
                    x_off_to_center_ll=-.5
                elif l.split("=")[1].strip().startswith("Right"):
                    y_off_to_center_ll=.5
                else:
                    raise Exception(f"What is {l}?")
            if l.startswith("DieRefPoint="):
                dierefptx=float(l.split("=")[1].strip().split(',')[0])/1e3
                dierefpty=float(l.split("=")[1].strip().split(',')[1])/1e3
                # See below where centerdiex, centerdiey are assigned
                #assert l.split("=")[1].strip()=="0,0", \
                #    f"Need to adjust the logic for defining center if {l} is not '0,0'"
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
        f"Never ran this flatangle {flatangle} before. 270 is NotchRight, 90 is NotchLeft."


    if (origin,wafertestangle)==('LL',0):
        dietable=pd.DataFrame({'ProberDieX':[die[0] for die in dies],'ProberDieY':[die[1] for die in dies]})
        #centerdiex=np.median(dietable['ProberDieX'])
        #centerdiey=np.median(dietable['ProberDieY'])
        centerdiex=refdiex+x_off_to_center_ll
        centerdiey=refdiey+y_off_to_center_ll
        x=dietable['ProberDieX']-centerdiex+dierefptx/xindex
        y=dietable['ProberDieY']-centerdiey+dierefpty/yindex
    elif (origin,wafertestangle)==('UL',0): # ... not validated
        logger.warning(f"Origin {origin}, WaferTestAngle {wafertestangle} has not been validated")
        dietable=pd.DataFrame({'ProberDieX':[die[0] for die in dies],'ProberDieY':[die[1] for die in dies]})
        #centerdiex=np.median(dietable['ProberDieX'])
        #centerdiey=np.median(dietable['ProberDieY'])
        centerdiex=refdiex+x_off_to_center_ll+dierefptx/xindex
        centerdiey=refdiey-y_off_to_center_ll-dierefpty/yindex
        x=dietable['ProberDieX']-centerdiex
        y=-(dietable['ProberDieY']-centerdiey)
    elif (origin,wafertestangle)==('UR',270):
        #logger.debug('Flipped XY because of origin/testangle')
        # Because of wafertestangle I think, Velox flips the order in which X and Y appear in the die list
        dietable=pd.DataFrame({'ProberDieX':[die[1] for die in dies],'ProberDieY':[die[0] for die in dies]})
        #centerdiex=np.median(dietable['ProberDieX'])
        #centerdiey=np.median(dietable['ProberDieY'])
        centerdiex=refdiex-x_off_to_center_ll-dierefptx/xindex
        centerdiey=refdiey-y_off_to_center_ll-dierefpty/yindex
        # And, because of origin and wafer testangle, -Y is up and -X is right
        x=-(dietable['ProberDieY']-centerdiey)
        y=-(dietable['ProberDieX']-centerdiex)
    else:
        raise Exception(f"What is {origin} with {wafertestangle}?")
    # If notch right, flip both signs
    if flatangle==270:
        #print('Flipped XY sign because this is notch right')
        x=-x
        y=-y

    # If die pattern is symmetric, then the median is probably supposed to be at wafer center
    meddiex=np.median(dietable['ProberDieX'])
    meddiey=np.median(dietable['ProberDieY'])
    if np.isclose(np.round(meddiex-.5),meddiex-.5):
        assert np.isclose(meddiex,centerdiex), f"Y Median is {meddiex} but Y Center is {centerdiex}"
    if np.isclose(np.round(meddiey-.5),meddiey-.5):
        assert np.isclose(meddiey,centerdiey), f"Y Median is {meddiey} but Y Center is {centerdiey}"

    dietable['ProberDieLoc']=pd.Series([f"x{dlx}y{dly}" for dlx,dly in zip(dietable['ProberDieX'],dietable['ProberDieY'])],dtype="string")
    dietable['DieRadius [mm]']=np.round(np.sqrt((x*xindex)**2+(y*yindex)**2),decimals=1)
    dietable['DieCenterX [mm]']=x*xindex
    dietable['DieCenterY [mm]']=y*yindex
    x-=.5
    y-=.5
    x=np.asarray(np.round(x),dtype='int')
    y=np.asarray(np.round(y),dtype='int')
    dietable['DieXY']=pd.Series([f"L{xi:+d}B{yi:+d}" for xi,yi in zip(x,y)],dtype='string')

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
            #with open(self.DIEMAP_DIR/f"{mask}_Diemap-info.pkl",'rb') as f:
            #    self._dielbs[mask]=pickle.load(f)
            self._dielbs[mask]=get_database().get_mask_info(mask)
        return self._dielbs[mask]
