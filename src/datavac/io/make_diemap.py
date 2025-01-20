"""Function for making a full-wafer diemap.

Author: Samuel James Bader
Contact: samuel.james.bader@gmail.com
Revision: 2024-11-12

"""
import pickle
from typing import Callable
from pathlib import Path
import pandas as pd
import numpy as np

def make_fullwafer_diemap(name:str, xindex:float, yindex:float, xoffset:float = 0, yoffset:float = 0,
                          radius:float=150, notchsize:float=5, discard_ratio:float = .3, plot:bool = False,
                          save_csv:bool = True, save_dir:Path=None,
                          labeller:Callable[[float,float],str] = (lambda x,y: f"{x:+d},{y:+d}")):
    """Produces a wafermap (ie set of points for each die) in the formats used by JMP or DataVacuum.

    By default, the x-direction is along the notch, increasing away from the notch, and
    the y-direction is perpendicular to notch, increasing upwards when the notch is left.
    Those are the definitions for interpretting xindex, yindex, xoffset and yoffset; however, since
    the labelling can be customized by supplying a labeller function, the actual labels assigned to
    each die can be chosen with any desired convention.  The labeller function takes two arguments,
    the x and y according to the above convention, and returns a string of whatever the user prefers
    to call that die.  The default is of the form "{x},{y}", but one may, for instance, shift the
    dies and invert the y-definition by saying "{x+1},{-y}", etc.

    If save_csv is True, csv files will be produced corresponding to the format used by JMP, one
    called "...-Name.csv" and one called "...-XY.csv".  However, to use these as JMP wafermaps,
    it is still necessary to open these in JMP, and save them as .jmp tables, and set a Map Role
    on the DieXY column in "...-Name.jmp" indicating "Shape Name Definition".

    For more information on JMP mapfiles, see here (JMP 16):
    https://www.jmp.com/support/help/en/16.2/?os=win&source=application#page/jmp/custom-map-files.shtml

    Args:
        name: Name of the wafermap to be generated.  Only used by the plotting/saving.
        xindex: length in mm of the die in the x (ie along-notch) direction
        yindex: length in mm of the die in the y (ie perp-to-notch) direction.
        xoffset: x-coordinate of the bottom-left-when-notchleft corner of the reference die
        yoffset: y-coordinate of the bottom-left-when-notchleft corner of the reference die
        radius: radius in mm of the wafer
        notchsize: size in mm of the notch indentation [not real notch, just for visual purposes]
        discard_ratio: dies which fit into a rectangle with area less than discard_ratio*xindex*yindex are deleted
        plot: whether to draw a matplotlib figure representing the wafer
        save_csv: produce the JMP-friendly .csv outputs
        save_dir: path where to save the csv or pickle files (defaults to current directory)
        labeller: label-making function which takes the x,y of a die and returns a die name

    Returns:
        (1) A Pandas table of die coordinates for simple mapping and (2) a dictionary of more complete geometric
        information about each die for graphical rendering
    """
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
                    for xi in range(0,int(round(np.ceil((radius+abs(xoffset))/xindex)+1)))
                    for yi in range(0,int(round(np.ceil((radius+abs(yoffset))/yindex)+1)))
                    for d in [(xi,yi),(xi,-yi),(-xi,yi),(-xi,-yi)]]
    plausible_dies=list(set(plausible_dies))
    x_lefts={}
    y_bottoms={}

    # For each possible one
    for diei,(diex,diey) in enumerate(plausible_dies):

        # The label
        label=labeller(diex,diey)

        # The x,y of the bottom left corner
        x,y=(diex)*xindex+xoffset,(diey)*yindex+yoffset

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
            complete_dies.append(label)

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
        if w*h<discard_ratio*xindex*yindex: continue

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
        if plot:
            plt.plot(*zip(*(rect_points+[rect_points[0]])),color='b')
            plt.plot(*zip(*(refined_poly_points_dodge_notch+[refined_poly_points_dodge_notch[0]])),color='r')
            plt.text(np.mean([xy[0] for xy in rect_points]),np.mean([xy[1] for xy in rect_points]),
                     label,ha='center',va='center',fontsize=6,color=('g' if label in complete_dies else 'k'))

        # Add to collection
        allmappoints[label]=refined_poly_points_dodge_notch
        x_lefts[label]=x
        y_bottoms[label]=y

    # Full geometric info for each die, useful for rendering graphical die-maps
    die_geometries={'xindex':xindex,'yindex':yindex,
                    'diameter': 2*radius, 'valid_dies': set(allmappoints.keys()),
                    'complete_dies': complete_dies,
                    'left_x':x_lefts,
                    'bottom_y':y_bottoms,
                    'patch_table': pd.DataFrame({
                        'DieXY': list(allmappoints.keys()),
                        'x': [[v[0] for v in lst] for lst in allmappoints.values()],
                        'y': [[v[1] for v in lst] for lst in allmappoints.values()],
                    }).set_index('DieXY').sort_index()}

    # Table of die coordinates
    die_coords=pd.DataFrame({
        'DieXY':list(allmappoints.keys()),
        'DieCenterX [mm]':np.array(list(x_lefts.values()))+.5*xindex,
        'DieCenterY [mm]':np.array(list(y_bottoms.values()))+.5*yindex,
        'DieRadius [mm]':np.asarray(np.round(np.sqrt((np.array(list(x_lefts.values()))+.5*xindex)**2+(np.array(list(y_bottoms.values()))+.5*yindex)**2)),dtype=int)
    })

    # Add a shape for the circle as well
    thetas=np.linspace(np.arctan2(-notchsize,-radius),np.arctan2(notchsize,-radius),60,endpoint=True)
    allmappoints["Circle"]=list(zip(radius*np.cos(thetas),radius*np.sin(thetas)))+[(-radius+notchsize,0)]


    # Output in CSVs in the form that JMP will like
    if save_csv:
        for notch,sgn1,ind1,sgn2,ind2 in [('Left',1,0,1,1),('Down',-1,1,1,0),('Right',-1,0,-1,1),('Up',1,1,-1,0)]:
            pd.DataFrame(dict(
                zip(["Shape ID","DieXY"], \
                    zip(*enumerate(allmappoints.keys(),start=1))))) \
                .to_csv(save_dir/f"{name}_Notch{notch}Diemap-Name.csv",index=False)
            pd.DataFrame(dict(
                zip(["Shape ID","Part ID","X","Y"], \
                    zip(*[(i,i,sgn1*mp[ind1],sgn2*mp[ind2])
                          for i,mps in enumerate(allmappoints.values(),start=1) for mp in mps])))) \
                .to_csv(save_dir/f"{name}_Notch{notch}Diemap-XY.csv",index=False)

    return die_coords,die_geometries
