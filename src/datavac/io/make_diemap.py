from typing import Callable, Union, Optional
from pathlib import Path
import pandas as pd
import numpy as np

def make_fullwafer_diemap(name:str, aindex:float, bindex:float, aoffset:float = 0, boffset:float = 0,
                          adim:Optional[float]=None, bdim:Optional[float]=None,
                          radius:float=150, notchsize:float=5, discard_ratio:float = .3, plot:bool = False,
                          save_csv:bool = True, save_dir:Path=None,
                          transform:Callable[[float,float],tuple[float,float]]=lambda x,y: (x,y),
                          labeller:str = "{x:+d},{y:+d}", discard_labels:tuple[str]=()):
    """Produces a wafermap (ie set of points for each die) in the formats used by JMP or DataVacuum.

    The a-direction is along the notch, increasing away from the notch, and
    the b-direction is perpendicular to notch, increasing upwards when the notch is left.
    Those are the definitions for interpretting aindex, bindex, aoffset and boffset; however, a
    transform function can map from (A,B) to (X,Y), with any origin/shift convention the user desires.

    If save_csv is True, csv files will be produced corresponding to the format used by JMP, one
    called "...-Name.csv" and one called "...-XY.csv".  However, to use these as JMP wafermaps,
    it is still necessary to open these in JMP, and save them as .jmp tables, and set a Map Role
    on the DieXY column in "...-Name.jmp" indicating "Shape Name Definition".

    For more information on JMP mapfiles, see here (JMP 16):
    https://www.jmp.com/support/help/en/16.2/?os=win&source=application#page/jmp/custom-map-files.shtml

    Args:
        name: Name of the wafermap to be generated.  Only used by the plotting/saving.
        aindex: length in mm of the die in the a (ie along-notch) direction
        bindex: length in mm of the die in the b (ie perp-to-notch) direction.
        aoffset: a-coordinate of the bottom-left-when-notchleft corner of the reference (A,B)=(0,0) die (default 0)
        boffset: b-coordinate of the bottom-left-when-notchleft corner of the reference (A,B)=(0,0) die (default 0)
        adim: the length along a of the die (defaults to aindex)
        bdim: the length along b of the die (defaults to bindex)
        radius: radius in mm of the wafer
        notchsize: size in mm of the notch indentation [not real notch, just for visual purposes]
        discard_ratio: dies which fit into a rectangle with area less than discard_ratio*adim*bdim are deleted
        plot: whether to draw a matplotlib figure representing the wafer
        save_csv: produce the JMP-friendly .csv outputs
        save_dir: path where to save the csv or pickle files (defaults to current directory)
        transform: function which takes a tuple (a,b) and returns a user-desired tuple (x,y)
        labeller: label-making template string which takes the x,y of a die and returns a die name
        discard_labels: labels to discard if present (useful for removing e.g. seats known to be over waferscribe)

    Returns:
        (1) A Pandas table of die coordinates for simple mapping and (2) a dictionary of more complete geometric
        information about each die for graphical rendering
    """

    # labeller will be used as a function below.  If it's a string, we'll make it a function.
    if type(labeller) is str: labeller= lambda x,y,lblstr=labeller: lblstr.format(x=x,y=y)

    # adim and bdim default to aindex and bindex
    if adim is None: adim=aindex
    if bdim is None: bdim=bindex

    # Default to current directory if not supplied
    save_dir=save_dir or Path.cwd()

    # This will map labels to coordinates of the die polygons
    allmappoints={}

    # This will list only the dies which are complete rectangles
    complete_dies=[]

    # Draw the circle
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import patches
        plt.figure()
        plt.xlim(-radius-notchsize,radius+notchsize)
        plt.ylim(-radius-notchsize,radius+notchsize)
        circ=patches.Circle((0,0),radius,facecolor='None',edgecolor='k',clip_on=False)
        plt.gca().add_patch(circ)
        plt.axis('square')
        plt.axis('off')

    # An exhaustive list of dies, some of which are outside the circle
    plausible_dies=[d
                    for ai in range(0,int(round(np.ceil((radius+abs(aoffset))/aindex)+1)))
                    for bi in range(0,int(round(np.ceil((radius+abs(boffset))/bindex)+1)))
                    for d in [(ai,bi),(ai,-bi),(-ai,bi),(-ai,-bi)]]
    plausible_dies=list(set(plausible_dies))

    # Note: terms like "left" and "bottom" are interpreted notch-left
    # while x and y are the user-transformed coordinates
    a_lefts={}
    b_bottoms={}
    diexs={}
    dieys={}

    # For each possible one
    for diei,(diea,dieb) in enumerate(plausible_dies):

        # The label
        diex,diey=transform(diea,dieb)
        label=labeller(diex,diey)
        if label in discard_labels: continue

        # The a,b of the bottom left corner
        a,b=(diea)*aindex+aoffset,(dieb)*bindex+boffset

        # All points of the die rectangle, going CW
        rect_points=[(a,b),(a,b+bdim),(a+adim,b+bdim),(a+adim,b)]

        # Get all the points of the rectangle and points where it intersects the circle, going CW
        poly_points=[]

        # For each segment ai,ai -> bf,bf, with 'h' or 'v' orientation
        for ((ai,bi),(af,bf)),orient in zip(zip(rect_points,(rect_points[1:]+[rect_points[0]])),['v','h','v','h']):

            # Check which points are inside circle
            i_inside=np.sqrt(ai**2+bi**2)<radius
            f_inside=np.sqrt(af**2+bf**2)<radius

            # If first is inside, add to our list
            if i_inside:
                poly_points.append((ai,bi))

            # If segment intersects circle (ie exactly one point is inside), include point of intersection
            if int(i_inside) + int(f_inside) == 1:
                if orient=='h':
                    ao,bo=np.sqrt(radius**2-min(bi**2,bf**2)),bi
                    ao=next((xoi for xoi in [ao,-ao] if (ai<=xoi<=af) or (af<=xoi<=ai)))
                    poly_points.append((ao,bo))
                elif orient=='v':
                    ao,bo=ai,np.sqrt(radius**2-min(ai**2,af**2))
                    bo=next((yoi for yoi in [bo,-bo] if (bi<=yoi<=bf) or (bf<=yoi<=bi)))
                    poly_points.append((ao,bo))

        # If no points collected, this is invalid die
        if not len(poly_points):
            continue

        # If the points are the unchanged, it's a complete die
        if rect_points==poly_points:
            complete_dies.append(label)

        # Now we'll add the arcs in
        refined_poly_points=[]

        # Go through every segment of the polygons we just created
        for (ai,bi),(af,bf) in zip(poly_points,(poly_points[1:]+[poly_points[0]])):

            # Add first point to list
            refined_poly_points.append((ai,bi))

            # If it's a vertical or horizontal segment, no more work needed
            if ai==af or bi==bf:
                continue

            # Otherwise, we'll add some points along the arc bounded by xi,yi and xf,yf
            else:
                theta1,theta2=np.arctan2(bi,ai),np.arctan2(bf,af)
                if theta2-theta1>np.pi: theta2-=2*np.pi
                if theta1-theta2>np.pi: theta1-=2*np.pi
                thetas=np.linspace(theta1,theta2,4,endpoint=False)
                refined_poly_points+=zip(radius*np.cos(thetas),radius*np.sin(thetas))

        # Find the area of a rectangle that would contain this polygon and if it's tiny, disregard die
        # Special case: if discard_ratio is 1, base on complete_dies list to avoid floating-point equality check
        w=max([ab[0] for ab in refined_poly_points])-min([ab[0] for ab in refined_poly_points])
        h=max([ab[1] for ab in refined_poly_points])-min([ab[1] for ab in refined_poly_points])
        if discard_ratio==1:
            if label not in complete_dies: continue
        else:
            if w*h<discard_ratio*adim*bdim: continue

        # Finally, remove the notch
        refined_poly_points_dodge_notch=[]
        for ap,bp in refined_poly_points:

            # If point is nowhere near notch, no worries, just add to list
            if (ap>-radius+notchsize) or abs(bp)>notchsize:
                refined_poly_points_dodge_notch.append((ap,bp))

            # If point is near notch and is "the" notchpoint, pull it in by notchsize
            elif bp==0:
                refined_poly_points_dodge_notch.append((ap+notchsize,bp))

        # Plot the die rectangle and the cropped die and annotate
        if plot:
            plt.plot(*zip(*(rect_points+[rect_points[0]])),color='b')
            plt.plot(*zip(*(refined_poly_points_dodge_notch+[refined_poly_points_dodge_notch[0]])),color='r')
            plt.text(np.mean([ab[0] for ab in rect_points]),np.mean([ab[1] for ab in rect_points]),
                     label,ha='center',va='center',fontsize=6,color=('g' if label in complete_dies else 'k'))

        # Add to collection
        allmappoints[label]=refined_poly_points_dodge_notch
        a_lefts[label]=a
        b_bottoms[label]=b
        diexs[label]=diex
        dieys[label]=diey

    # Full geometric info for each die, useful for rendering graphical die-maps
    die_geometries={'xindex':aindex,'yindex':bindex,
                    'diameter': 2*radius, 'valid_dies': set(allmappoints.keys()),
                    'complete_dies': complete_dies,'notchsize':notchsize,
                    'left_x':a_lefts,
                    'bottom_y':b_bottoms,
                    'patch_table': pd.DataFrame({
                        'DieXY': list(allmappoints.keys()),
                        'x': [[v[0] for v in lst] for lst in allmappoints.values()],
                        'y': [[v[1] for v in lst] for lst in allmappoints.values()],
                    }).set_index('DieXY').sort_index()}

    # Table of die coordinates
    a_diecenters=np.array(list(a_lefts.values()))+.5*adim
    b_diecenters=np.array(list(b_bottoms.values()))+.5*bdim
    die_coords=pd.DataFrame({
        'DieXY':list(allmappoints.keys()),
        'DieX':list(diexs.values()),
        'DieY':list(dieys.values()),
        'DieCenterA [mm]':a_diecenters,
        'DieCenterB [mm]':b_diecenters,
        'DieRadius [mm]':np.asarray(np.round(np.sqrt((np.array(list(a_lefts.values()))+.5*adim)**2
                                                    +(np.array(list(b_bottoms.values()))+.5*bdim)**2)),dtype=int),
        'DieComplete':[(k in complete_dies) for k in allmappoints.keys()],
    })

    # Add a shape for the circle as well
    thetas=np.linspace(np.arctan2(-notchsize,-radius),np.arctan2(notchsize,-radius),60,endpoint=True)
    allmappoints["Circle"]=list(zip(radius*np.cos(thetas),radius*np.sin(thetas)))+[(-radius+notchsize,0)]
    diexs['Circle']=pd.NA; dieys['Circle']=pd.NA # needed because of use of zip function for *Name.csv


    # Output in CSVs in the form that JMP will like
    if save_csv:
        (save_dir:=Path(save_dir)).mkdir(exist_ok=True,parents=True)
        for notch,sgn1,ind1,sgn2,ind2 in [('Left',1,0,1,1),('Down',-1,1,1,0),('Right',-1,0,-1,1),('Up',1,1,-1,0)]:
            if (save_csv!=True) and (notch not in save_csv): continue
            pd.DataFrame(dict(
                zip(["Shape ID","DieXY","DieX","DieY"], \
                    [*zip(*enumerate(allmappoints.keys(),start=1)),diexs.values(),dieys.values()]))) \
                .to_csv(save_dir/f"{name}_Notch{notch}Diemap-Name.csv",index=False)
            pd.DataFrame(dict(
                zip(["Shape ID","Part ID","X","Y"], \
                    zip(*[(i,i,sgn1*mp[ind1],sgn2*mp[ind2])
                          for i,mps in enumerate(allmappoints.values(),start=1) for mp in mps])))) \
                .to_csv(save_dir/f"{name}_Notch{notch}Diemap-XY.csv",index=False)

    return die_coords,die_geometries


def generate_custom_remap(main:pd.DataFrame,
                          main1xy:tuple[int,int], custom1xy:tuple[int,int],
                          main2xy:tuple[int,int], custom2xy:tuple[int,int],
                          main3xy:tuple[int,int], custom3xy:tuple[int,int]) -> pd.DataFrame:
    """ Generate a transform table from main to custom.

    Args:
        main (DataFrame): a table containing DieX (int), DieY (int), DieXY (str) for the main
            coordinate system (such as that output by make_fullwafer_diemap)
        main(1/2/3)xy (tuple of ints): (x,y) pairs for three dies in main coordinates.  The three
            dies should form an 'L' shape with main2xy at the juncture, ie of the other two points
            one differs from main2xy only in y and the other differs from main2xy only in x.
        custom(1/2/3)xy (tuple of ints): (x,y) pairs for same three dies in custom coordinates.

    Returns:
        a Pandas table with CustomDieX, CustomDieY and the main DieX, DieY, DieXY they map to.
    """

    # Reorder such that the first two points differ along 'y' for main
    if main1xy[1]!=main2xy[1]:
        c1x,c1y=custom1xy; c2x,c2y=custom2xy; c3x,c3y=custom3xy
        m1x,m1y=main1xy  ; m2x,m2y=main2xy  ; m3x,m3y=main3xy
    else:
        c1x,c1y=custom3xy; c2x,c2y=custom2xy; c3x,c3y=custom1xy
        m1x,m1y=main3xy  ; m2x,m2y=main2xy  ; m3x,m3y=main1xy

    # Determine the rotation part of the transform
    xyflip=(c1y==c2y)
    sgnymain = int(np.sign((c2x-c1x)/(m2y-m1y) if xyflip else (c2y-c1y)/(m2y-m1y)))
    sgnxmain = int(np.sign((c3y-c2y)/(m3x-m2x) if xyflip else (c3x-c2x)/(m3x-m2x)))

    # Determine the shift part of the transform
    isgnymain, isgnxmain = sgnymain, sgnxmain
    affx,affy = (c2y*isgnxmain-m2x,c2x*isgnymain-m2y) if xyflip else (c2x*isgnxmain-m2x,c2y*isgnymain-m2y)

    # Transform the given points to check
    trans = lambda mx,my: ((my+affy)*sgnymain,(mx+affx)*sgnxmain) if xyflip else ((mx+affx)*sgnxmain,(my+affy)*sgnymain)
    assert trans(*main1xy)==tuple(custom1xy), (tuple(custom1xy),trans(*main1xy),main1xy)
    assert trans(*main2xy)==tuple(custom2xy), (tuple(custom2xy),trans(*main2xy),main2xy)
    assert trans(*main3xy)==tuple(custom3xy), (tuple(custom3xy),trans(*main3xy),main3xy)

    # Transform everything
    return pd.DataFrame(dict(       zip([     'CustomDieX', 'CustomDieY',     'DieX',    'DieY',     'DieXY',     'DieCenterA [mm]',     'DieCenterB [mm]'],
                                 zip(*[(*trans(row['DieX'], row['DieY']),row['DieX'],row['DieY'],row['DieXY'],row['DieCenterA [mm]'],row['DieCenterB [mm]'])
                                                           for _, row in main.iterrows()])))).convert_dtypes()