import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

#import matplotlib.pyplot as plt
#from matplotlib import patches

import pickle

from datavac.io.database import get_database
from datavac.logging import logger
from datavac.util import check_dtypes

def make_fullwafer_diemap(name, xindex, yindex, radius=150, notchsize=5, plot=False, save=True):
    # This was originally written with a convenient matplotlib view but matplotlib is not
    # listed as a project requirement at the moment, so suppressing plotting functionality for now.
    # TODO: many of the names in this function don't reflect other usage in the code (DieLB, UDL, etc)
    assert plot==False

    # This will map UDL (Universal Die Labels) to coordinates of the die polygons
    allmappoints={}

    # This will list only the dies which are complete rectangles
    complete_dies=[]

    # Draw the circle
    #if plot:
    #    plt.figure()
    #    plt.xlim(-radius-notchsize,radius+notchsize)
    #    plt.ylim(-radius-notchsize,radius+notchsize)
    #    circ=patches.Circle((0,0),radius,facecolor='None',edgecolor='k',clip_on=False)
    #    plt.gca().add_patch(circ)
    #    plt.axis('square')
    #    plt.axis('off')

    # An exhaustive list of dies, some of which are outside the circle
    plausible_dies=[d
                    for xi in range(0,int(round(np.ceil(radius/xindex)+1)))
                    for yi in range(0,int(round(np.ceil(radius/yindex)+1)))
                    for d in [(xi,yi),(xi,-yi),(-xi,yi),(-xi,-yi)]]
    plausible_dies=list(set(plausible_dies))
    x_lefts={}
    y_bottoms={}

    # For each possible one
    for diei,(udlx,udly) in enumerate(plausible_dies):

        # The UDL
        udl=f"L{udlx:+d}B{udly:+d}"

        # The x,y of the bottom left corner
        x,y=(udlx)*xindex,(udly)*yindex

        # All points of the die rectangle, going CW
        rect_points=[(x,y),(x,y+yindex),(x+xindex,y+yindex),(x+xindex,y)]

        # Get all the points of the rectangle and points where it intersects the circle, going CW
        poly_points=[]

        # For each segment xi,yi -> xf,yf, with 'h' or 'v' orientation
        for ((xi,yi),(xf,yf)),orient in zip(zip(rect_points,(rect_points[1:]+[rect_points[0]])),['v','h','v','h']):

            # Check which points are inside circle
            i_inside=np.sqrt(xi**2+yi**2)<radius
            f_inside=np.sqrt(xf**2+yf**2)<radius

            # If first is inside, add to our list
            if i_inside:
                poly_points.append((xi,yi))

            # If segment intersects circle (ie exactly one point is inside), include point of intersection
            if int(i_inside) + int(f_inside) == 1:
                if orient=='h':
                    xo,yo=np.sqrt(radius**2-min(yi**2,yf**2)),yi
                    xo=next((xoi for xoi in [xo,-xo] if (xi<=xoi<=xf) or (xf<=xoi<=xi)))
                    poly_points.append((xo,yo))
                elif orient=='v':
                    xo,yo=xi,np.sqrt(radius**2-min(xi**2,xf**2))
                    yo=next((yoi for yoi in [yo,-yo] if (yi<=yoi<=yf) or (yf<=yoi<=yi)))
                    poly_points.append((xo,yo))

        # If no points collected, this is invalid die
        if not len(poly_points):
            continue

        # If the points are the unchanged, it's a complete die
        if rect_points==poly_points:
            complete_dies.append(udl)

        # Now we'll add the arcs in
        refined_poly_points=[]

        # Go through every segment of the polygons we just created
        for (xi,yi),(xf,yf) in zip(poly_points,(poly_points[1:]+[poly_points[0]])):

            # Add first point to list
            refined_poly_points.append((xi,yi))

            # If it's a vertical or horizontal segment, no more work needed
            if xi==xf or yi==yf:
                continue

            # Otherwise, we'll add some points along the arc bounded by xi,yi and xf,yf
            else:
                theta1,theta2=np.arctan2(yi,xi),np.arctan2(yf,xf)
                if theta2-theta1>np.pi: theta2-=2*np.pi
                if theta1-theta2>np.pi: theta1-=2*np.pi
                thetas=np.linspace(theta1,theta2,4,endpoint=False)
                refined_poly_points+=zip(radius*np.cos(thetas),radius*np.sin(thetas))

        # Find the area of a rectangle that would contain this polygon and if it's tiny, disregard die
        w=max([xy[0] for xy in refined_poly_points])-min([xy[0] for xy in refined_poly_points])
        h=max([xy[1] for xy in refined_poly_points])-min([xy[1] for xy in refined_poly_points])
        if w*h<.3*xindex*yindex: continue

        # Finally, remove the notch
        refined_poly_points_dodge_notch=[]
        for xp,yp in refined_poly_points:

            # If point is nowhere near notch, no worries, just add to list
            if (xp>-radius+notchsize) or abs(yp)>notchsize:
                refined_poly_points_dodge_notch.append((xp,yp))

            # If point is near notch and is "the" notchpoint, pull it in by notchsize
            elif yp==0:
                refined_poly_points_dodge_notch.append((xp+notchsize,yp))

        # Plot the die rectangle and the cropped die and annotate
        #if plot:
        #    plt.plot(*zip(*(rect_points+[rect_points[0]])),color='b')
        #    plt.plot(*zip(*(refined_poly_points_dodge_notch+[refined_poly_points_dodge_notch[0]])),color='r')
        #    plt.text(np.mean([xy[0] for xy in rect_points]),np.mean([xy[1] for xy in rect_points]),
        #             udl,ha='center',va='center',fontsize=6,color=('g' if udl in complete_dies else 'k'))

        # Add to collection
        allmappoints[udl]=refined_poly_points_dodge_notch
        x_lefts[udl]=x
        y_bottoms[udl]=y

    # Dump the mapping info
    die_info_to_pickle={'xindex':xindex,'yindex':yindex,
                         'diameter': 2*radius, 'valid_dies': set(allmappoints.keys()),
                         'complete_dies': complete_dies,
                         'left_x':x_lefts,
                         'bottom_y':y_bottoms,
                         'patch_table': pd.DataFrame({
                             'DieXY': list(allmappoints.keys()),
                             'x': [[v[0] for v in lst] for lst in allmappoints.values()],
                             'y': [[v[1] for v in lst] for lst in allmappoints.values()],
                         }).set_index('DieXY').sort_index()
                         }
    if save:
        with open(project.DIEMAP_DIR/f"{name}_Diemap-info.pkl",'wb') as f:
            pickle.dump(die_info_to_pickle,f)

    # To put in the database
    dbdf=pd.DataFrame({
        'DieXY':list(allmappoints.keys()),
        'DieRadius [mm]':np.asarray(np.round(np.sqrt((np.array(list(x_lefts.values()))+.5*xindex)**2+(np.array(list(y_bottoms.values()))+.5*yindex)**2)),dtype=int)
    })

    # Add a shape for the circle as well
    thetas=np.linspace(np.arctan2(-notchsize,-radius),np.arctan2(notchsize,-radius),60,endpoint=True)
    allmappoints["Circle"]=list(zip(radius*np.cos(thetas),radius*np.sin(thetas)))+[(-radius+notchsize,0)]


    # Output in CSVs in the form that JMP will like
    if save:
        pd.DataFrame(dict(
            zip(["Shape ID","DieXY"], \
                zip(*enumerate(allmappoints.keys(),start=1))))) \
            .to_csv(project.DIEMAP_DIR/f"{name}_Diemap-Name.csv",index=False)
        pd.DataFrame(dict(
            zip(["Shape ID","Part ID","X","Y"], \
                zip(*[(i,i,mp[0],mp[1]) for i,mps in enumerate(allmappoints.values(),start=1) for mp in mps])))) \
            .to_csv(project.DIEMAP_DIR/f"{name}_Diemap-XY.csv",index=False)

    if save:
        pd.DataFrame(dict(
            zip(["Shape ID","DieXY"], \
                zip(*enumerate(allmappoints.keys(),start=1))))) \
            .to_csv(project.DIEMAP_DIR/f"{name}_NotchDownDiemap-Name.csv",index=False)
        pd.DataFrame(dict(
            zip(["Shape ID","Part ID","X","Y"], \
                zip(*[(i,i,-mp[1],mp[0]) for i,mps in enumerate(allmappoints.values(),start=1) for mp in mps])))) \
            .to_csv(project.DIEMAP_DIR/f"{name}_NotchDownDiemap-XY.csv",index=False)

    return dbdf,die_info_to_pickle

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
        dietable=pd.DataFrame({'ProberDieX':[die[0] for die in dies],'ProberDieY':[die[1] for die in dies]})
        centerdiex=np.median(dietable['ProberDieX'])
        centerdiey=np.median(dietable['ProberDieY'])
        x=dietable['ProberDieX']-centerdiex
        y=dietable['ProberDieY']-centerdiey
    if (origin,wafertestangle)==('UL',0): # ... not validated
        logger.warning(f"Origin {origin}, WaferTestAngle {wafertestangle} has not been validated")
        dietable=pd.DataFrame({'ProberDieX':[die[0] for die in dies],'ProberDieY':[die[1] for die in dies]})
        centerdiex=np.median(dietable['ProberDieX'])
        centerdiey=np.median(dietable['ProberDieY'])
        x=dietable['ProberDieX']-centerdiex
        y=-(dietable['ProberDieY']-centerdiey)
    if (origin,wafertestangle)==('UR',270):
        #logger.debug('Flipped XY because of origin/testangle')
        # Because of wafertestangle I think, Velox flips the order in which X and Y appear in the die list
        dietable=pd.DataFrame({'ProberDieX':[die[1] for die in dies],'ProberDieY':[die[0] for die in dies]})
        centerdiex=np.median(dietable['ProberDieX'])
        centerdiey=np.median(dietable['ProberDieY'])
        # And, because of origin and wafer testangle, -Y is up and -X is right
        x=-(dietable['ProberDieY']-centerdiey)
        y=-(dietable['ProberDieX']-centerdiex)
    # If notch right, flip both signs
    if flatangle==270:
        #print('Flipped XY sign because this is notch right')
        x=-x
        y=-y

    assert np.isclose(np.round(centerdiex-.5),centerdiex-.5), f"Not sure how to find centerdiex {centerdiex}"
    assert np.isclose(np.round(centerdiey-.5),centerdiey-.5), f"Not sure how to find centerdiey {centerdiey}"

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
