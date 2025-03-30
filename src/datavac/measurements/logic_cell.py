import ast
from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import pandas as pd

from datavac.io.measurement_table import UniformMeasurementTable
from datavac.measurements.measurement_type import MeasurementType


@dataclass
class OscopeLogic(MeasurementType):
    formula_col: str = 'formula'
    vhi: float = 1
    vlo: float = 0
    channel_mapping:Optional[dict[str,Sequence[str]]]=None
    tcluster_thresh: float = None

    def analyze(self, measurements: UniformMeasurementTable):
        tstep=np.mean(np.diff(measurements['Time'],axis=1))
        truth_table_pass=pd.Series([pd.NA]*len(measurements),dtype='boolean')
        for formula, grp in measurements.scalar_table_with_layout_params([self.formula_col]).groupby(self.formula_col):
            # Hopefully temporary, but since "|" is used as a delimiter for CSV uploading,
            # it can't appear in the formula in the layout params table
            formula:str=formula.replace("%OR%","|")

            outside,inpside=formula.split("=")
            tree = ast.parse(inpside.strip())
            inp_names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
            out_names = outside.split(",")
            channel_mapping={h:[h] for h in measurements.headers} if self.channel_mapping is None else self.channel_mapping
            curves={name:measurements[ch][list(grp.index),:] for ch, names in channel_mapping.items()
                    for name in names if ch in measurements.headers
                                            and name in [n.split("_")[0] for n in inp_names+out_names]}

            vmid=(self.vhi+self.vlo)/2
            raw_bin_for_clusters= {k.split("_")[0]: curves[k.split("_")[0]]>vmid for k in inp_names+out_names}
            arng=np.vstack([np.arange(measurements['Time'].shape[1]-1) for i in range(len(measurements))])
            iflips=sorted([i for k in raw_bin_for_clusters for i in arng[np.diff(raw_bin_for_clusters[k])!=0]])
            iflips = [0,*iflips,measurements['Time'].shape[1]-1]

            # If a clustering threshold is specified in time, translate it to points
            if self.tcluster_thresh is not None:
                iclusterthresh = int(round(self.tcluster_thresh/tstep))
            # Otherwise calculate a good threshold as a fraction of the max interval between any flips
            else:
                iclusterthresh=int(round(np.max(np.diff(iflips))/4))

            clusters=[[iflips[0]]]
            for i in iflips:
                if i-clusters[-1][-1] > iclusterthresh: clusters.append([i])
                else: clusters[-1].append(i)
            iflips=[int(round(np.median(c))) for c in clusters]

            digitized_curves={k:np.vstack([np.mean(curves[k][:,imin:imax],axis=1)>vmid
                                     for imin,imax in zip(iflips[:-1],iflips[1:])]).T
                                for k in curves}

            #print(digitized_curves)

            inps = {}
            first_nocare=False
            for k in inp_names:
                if k.endswith("_n"):
                    inps[k] = digitized_curves[k.split("_")[0]]
                elif k.endswith("_nm1"):
                    inps[k] = np.roll(digitized_curves[k.split("_")[0]],shift=1,axis=1)
                    first_nocare=True
                else:
                    raise Exception(f"Input '{k}' should end with _n or _nm1")
            calc_outs = dict(zip([k.strip() for k in outside.split(",")],eval(inpside.strip(),{"__builtins__": None}, inps)))
            ttfails=[]
            for k in calc_outs:
                mask=slice(None) if not first_nocare else slice(1,None)
                assert k.endswith("_n"), f"How is there an output curve '{k}' that doesn't end with _n?"
                ttptfail= digitized_curves[k.split("_n")[0]][:,mask]!=calc_outs[k][mask]
                ttfails.append(np.sum(ttptfail,axis=1))

            truth_table_pass[grp.index]=(np.sum(ttfails,axis=0)==0)
        measurements['truth_table_pass']=truth_table_pass


