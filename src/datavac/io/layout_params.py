import os
import re
from datetime import datetime
from importlib import import_module
from pathlib import Path

from datavac.util.cli import cli_helper
from datavac.util.logging import logger, time_it
import pandas as pd
import numpy as np
import pickle
import traceback
import yaml

LAYOUT_PARAMS_DIR=Path(os.environ.get("DATAVACUUM_LAYOUT_PARAMS_DIR"))

# TODO: remove the LayoutParameters gotcha below and change this _LayoutParameters back to LayoutParameters
class _LayoutParameters:

    def __init__(self):
        self.regenerate_from_excel()

    def timestamp_still_valid(self):
        yaml_path = Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"layout_params.yaml"
        with open(yaml_path,'r') as f:
            current_yaml_str=f.read()

        if (yaml_path.stat().st_mtime>self._generated_timestamp) and self._yaml_str!=self._yaml:
            logger.info("layout_params.yaml has more recent timestamp than cache and doesn't match")
            return False
        else:
            layout_param_paths=self._yaml['layout_param_paths']
            outofdate_found=False
            for mask,paths_by_mask in layout_param_paths.items():
                for path in paths_by_mask:
                    if (LAYOUT_PARAMS_DIR/path).stat().st_mtime>self._generated_timestamp:
                        logger.info(f"LayoutParam cache is missing or out-of-date for {path}")
                        outofdate_found=True
            return not outofdate_found

    def regenerate_from_excel(self):
        yaml_path = Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"layout_params.yaml"
        with open(yaml_path,'r') as f:
            self._yaml_str=f.read()
            self._yaml=yaml.safe_load(self._yaml_str)
        self._cat_tables: dict[(str,str),pd.DataFrame] ={}
        self._dut_to_catkey: dict[(str,str),(str,str)] ={}
        self._tables_by_meas: dict[str,pd.DataFrame] ={}
        self._rows_by_mask: dict[str,list[str]] ={}
        self._generated_timestamp=datetime.now().timestamp()

        for mask,paths_by_mask in self._yaml['layout_param_paths'].items():
            self._rows_by_mask[mask]=[]
            for path in paths_by_mask:
                logger.info(f"Reading {str(LAYOUT_PARAMS_DIR/path)}")
                with open(LAYOUT_PARAMS_DIR/path,'rb') as f:
                    xls=pd.ExcelFile(f,engine='openpyxl')
                    for sh in xls.book.sheetnames:
                        if 'IGNORE' in sh:
                            #logger.debug(f"Ignoring '{sh}' because of its name")
                            continue
                        table=pd.read_excel(xls,sh)#
                        table['rowname']=table['rowname'].ffill().astype('string')
                        if not ('rowname' in table.keys() and 'DUT' in table.keys()):
                            logger.debug(f"Ignoring '{sh}' because missing rowname and/or DUT")
                            continue
                        self._rows_by_mask[mask]+=list(table['rowname'].unique())
                        table=table.rename(columns={c:c.replace("\n"," ").strip() for c in table.columns})
                        table.drop(columns=[c for c in self._yaml.get("drop_param_names",[]) if c in table],inplace=True)
                        table=table.rename(columns={k:v for k,v in self._yaml['replace_param_names'].items() if k in table.columns})
                        table['Structure']=(table['RowName']+"-DUT"+table['DUT'].astype(str).str.zfill(2)).astype('string')
                        table.set_index('Structure',inplace=True)
                        pad_cols=[col for col in table.columns if col[:4]=="PAD:"]
                        for c in pad_cols:
                            if table[c].dtype==np.dtype('O'):
                                table[c]=[int(p[3:]) for p in table[c]]
                        table=table.rename(columns={col:"PAD:"+col[4:].lower() for col in pad_cols})
                        table=table.convert_dtypes()
                        for c in table.columns:
                            if table[c].dtype==np.dtype('O'):
                                logger.debug(f"Stringifying '{c}' in '{sh}'")
                                table[c]=table[c].astype('string')
                        self._cat_tables[(mask,sh)]=table
                        for structure,r in table.iterrows():
                            self._dut_to_catkey[(structure,mask)]=(mask,sh)

        logger.info("Collating by measurement group")
        by_meas_group=self._yaml['by_meas_group']
        for meas_key in by_meas_group:
            self._tables_by_meas[meas_key]=pd.DataFrame({})
            sensible_column_order=[]
            for mask in self._yaml['layout_param_paths']:
                if mask in by_meas_group[meas_key]:
                    for catregex in by_meas_group[meas_key][mask]:
                        for _mask,cat in self._cat_tables:
                            if _mask==mask and re.match(catregex,cat):
                                self._tables_by_meas[meas_key]=\
                                    self._tables_by_meas[meas_key].combine_first(self._cat_tables[(mask,cat)])
                                im1=-1;
                                for c in self._cat_tables[(mask,cat)].columns:
                                    try: im1=sensible_column_order.index(c)
                                    except ValueError: sensible_column_order.insert((im1:=im1+1),c)
            assert set(self._tables_by_meas[meas_key].columns)==set(sensible_column_order)
            self._tables_by_meas[meas_key]=self._tables_by_meas[meas_key][sensible_column_order]

            if (afunc_dotpaths:=by_meas_group[meas_key].get('apply',None)):
                for afunc_dotpath in afunc_dotpaths:
                    afunc=getattr(import_module(afunc_dotpath.split(":")[0]),
                                  afunc_dotpath.split(":")[1])
                    afunc(self._tables_by_meas[meas_key])
            for c in self._tables_by_meas[meas_key].columns:
                if self._tables_by_meas[meas_key][c].dtype==np.dtype('O'):
                    logger.debug(f"Stringifying '{c}' in '{meas_key}'")
                    self._tables_by_meas[meas_key][c]=self._tables_by_meas[meas_key][c].astype('string')
            if 'names' in by_meas_group[meas_key]:
                tab_to_rename=self._tables_by_meas[meas_key]
                del self._tables_by_meas[meas_key]
                for altname in by_meas_group[meas_key]['names']:
                    self._tables_by_meas[altname]=tab_to_rename

    #def __getitem__(self,item):
    #    raise Exception("Didn't update this when I changed get_params to require mask")
    #    if type(item) is str:
    #        res=self.get_params([item])
    #        return res.iloc[0]
    #    else:
    #        return self.get_params(item)

    def _get_correction_map(self,structures,mask):
        correction_map={}
        for structure in structures:
            if (structure,mask) not in self._dut_to_catkey:
                supplied_row,dut=structure.split("-",maxsplit=1)
                corrected_row=self.search_partial_rowname(mask,supplied_row)
                dut=dut.replace("PAD","")
                if len(dut)!=5:
                    if dut.startswith("DUT"):
                        dut=int(dut[3:])
                    else:
                        dut=int(dut)
                    dut=f"DUT{dut:02d}"
                corrected_structure=corrected_row+'-'+dut
                correction_map[structure]=corrected_structure
        return correction_map


    def get_params(self,structures,mask,drop_pads=True,for_measurement_group=None,allow_partial=False):
        if allow_partial:
            correction_map=self._get_correction_map(structures,mask=allow_partial)
            structures=[correction_map.get(structure,structure) for structure in structures]

        if for_measurement_group:
            try:
                tab=self._tables_by_meas[for_measurement_group].loc[structures]
            except KeyError as e:
                raise KeyError(str(e)[1:-1]+f" when looking up within group '{for_measurement_group}'."\
                        "  The structure *is* found in the layout parameters, but not within this specific group.")
        else:
            rows=[]
            for structure in structures:
                rows.append(singular_row:=self._cat_tables[self._dut_to_catkey[(structure,mask)]].loc[[structure]])
                assert len(singular_row)==1
            tab=pd.concat(rows)

        if drop_pads:
            tab=tab.drop(columns=[c for c in tab.columns if c[:4]=="PAD:"])
        if allow_partial:
            return tab, correction_map
        else:
            return tab

    def search_partial_rowname(self,mask,partial_rowname):
        if partial_rowname in self._yaml['common_mistakes'].get(mask,{}):
            match=self._yaml['common_mistakes'][mask][partial_rowname]
            assert match in self._rows_by_mask[mask],\
                f"{match} (from {partial_rowname} in common mistakes list) not in {mask}!"
            return match

        else:
            for s in (reps:=self._yaml.get('common_replacements',{}).get(mask,{})):
                partial_rowname=partial_rowname.replace(s,reps[s])
            matches=[row for row in self._rows_by_mask[mask] if partial_rowname==row or
                     '_' in row and partial_rowname==row.split('_')[1]]
            if len(matches)==0:
                raise Exception(f"Row '{partial_rowname}' not found in {mask}")
            if len(matches)>1:
                if not all(matches[0]==row for row in matches):
                    raise Exception(f"Multiple possible matches {matches} for '{partial_rowname}' in {mask}")
            return matches[0]


    def regularize_structures(self,meas_df,mask):
        structures_to_get=meas_df['Structure'].unique()
        correction_map=self._get_correction_map(structures_to_get,mask)
        #logger.info(f"Correcting structure names automatically: {correction_map}")
        for structure in structures_to_get:
            if structure not in correction_map:
                correction_map[structure]=structure

        meas_df['Structure']=meas_df['Structure'].map(correction_map).astype('string')
        meas_df['RowName']=meas_df['Structure'].str.split("-",expand=True)[0]
        meas_df['Site']=meas_df['RowName']+\
                        "-"+meas_df['RowRep'].astype(str).str.zfill(2)+\
                        "-DUT"+meas_df['DUT'].astype(str).str.zfill(2)

    def validate_structures_in_meas_group(self,structures,meas_group):
        rejects=[structure for structure in structures if structure not in self._tables_by_meas[meas_group].index]
        assert len(rejects)==0, f"Structures {rejects} not in measurement parameter group '{meas_group}'"


    def merge_with_layout_params(self,meas_df,for_measurement_group,param_names=None, on_missing='error'):
        #structures_to_get=meas_df['Structure'].unique()
        #params=self.get_params(structures_to_get,allow_partial=False,for_measurement_group=for_measurement_group)
        if for_measurement_group not in self._tables_by_meas:
            raise Exception(f"No layout parameters for measurement group {for_measurement_group}")
        params=self._tables_by_meas[for_measurement_group]
        if param_names:
            params=params[[pn for pn in param_names if pn in params.columns]].copy()
            for param in param_names:
                if param not in params.columns:
                    if on_missing=='error':
                        raise Exception(f"Missing parameter {param}")
                    elif on_missing=='NA':
                        params[param]=pd.NA
                    elif on_missing=='ignore':
                        continue
                    else:
                        raise Exception(f"Unrecognized value for on_missing={on_missing}")
        else:
            cols_to_drop=[c for c in ['RowName','DUT'] if c in meas_df]+[c for c in params if c.startswith("PAD:")]
            params=params.drop(columns=cols_to_drop)
        merged=pd.merge(left=meas_df,right=params,how='left',left_on=['Structure'],right_index=True,
                        suffixes=(None,'_param'))
        #import pdb; pdb.set_trace()
        return merged

class LayoutParameters(_LayoutParameters):
    def __init__(self,*args,**kwargs):
        raise Exception("USE get_layout_parameters()")

_layout_params:'LayoutParameters'=None
_layout_params_timestamp:float=None
def get_layout_params(force_regenerate=False, conn=None):
    global _layout_params
    if force_regenerate or (_layout_params is None):
        from datavac.io.database import pickle_db_cached
        _layout_params,_layout_params_timestamp=pickle_db_cached('LayoutParams',namespace='vac',conn=conn)(_LayoutParameters)(force=force_regenerate)
    return _layout_params

def cli_layout_params_valid():
    from datavac.io.database import get_database
    db=get_database(populate_metadata=False)
    if get_layout_params().timestamp_still_valid():
        print("Layout params are valid")
    else:
        print("Layout params need to be regenerated (use 'datavac update_layout_params').")

cli_layout_params=cli_helper(cli_funcs={
    'check_layout_params_valid (clpv)': 'datavac.io.layout_params:cli_layout_params_valid',
    'update_layout_params (ulp)': 'datavac.io.database:cli_update_layout_params',
})
