import os
import re
from importlib import import_module
from pathlib import Path

from datavac.util.logging import logger
import pandas as pd
import numpy as np
import pickle
import traceback
import yaml


class LayoutParameters:
    _instance=None

    CACHED_PATH=Path(os.environ["DATAVACUUM_CACHE_DIR"])/"LayoutParams.pkl"
    LAYOUT_PARAMS_DIR=Path(os.environ["DATAVACUUM_LAYOUT_PARAMS_DIR"])

    # This singleton pattern gets really unwieldy, consider a factory_function instead...
    def __new__(cls,force_regenerate=False):

        # See if there's no singleton yet
        if cls._instance is None:

            # If there's no singleton, see if you need to regenerate (forced or cache-out-of-date)
            ytr=cls.yaml_to_regenerate(force_regenerate=force_regenerate)

            # If no need to regenerate, try unpickling and return the result
            if not ytr:
                try:
                    with open(cls.CACHED_PATH,'rb') as f:
                        cls._instance=super().__new__(cls)
                        cls._instance=pickle.load(f)
                except Exception as e:
                    logger.warning(''.join(traceback.format_exception(e)))
                    logger.warning("Trouble reading cache, will regenerate")
                    force_regenerate=True
            # If there is a need to regenerate (including failure of the above try)
            if force_regenerate or ytr:
                logger.info("Regenerating layout params")

                # We'll make a new instance and regenerate
                cls._instance=super().__new__(cls)
                if ytr is None:
                    ytr=cls.yaml_to_regenerate(force_regenerate=True)
                cls._instance.regenerate_from_excel(ytr)
        # If there is a singleton
        else:
            # Regenerate if required
            if force_regenerate:
                ytr=cls.yaml_to_regenerate(force_regenerate=force_regenerate)
                cls._instance.regenerate_from_excel(ytr)

        # Return singleton
        return cls._instance

    @classmethod
    def yaml_to_regenerate(cls,force_regenerate=False):

        #yaml_path= Path(os.environ["DATAVACUUM_LAYOUT_PARAMS_YAML"])
        yaml_path = Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"layout_params.yaml"
        cached_path=cls.CACHED_PATH

        cached_time:float= cached_path.stat().st_mtime if cached_path.exists() else -np.inf
        need_to_regenerate=force_regenerate
        if yaml_path.stat().st_mtime>cached_time:
            logger.info("layout_params.yaml has changed")
            need_to_regenerate=True

        with open(yaml_path,'r') as f:
            loaded_yaml=yaml.safe_load(f)

        if not need_to_regenerate:
            layout_param_paths=loaded_yaml['layout_param_paths']
            for mask,paths_by_mask in layout_param_paths.items():
                for path in paths_by_mask:
                    if not need_to_regenerate:
                        if (cls.LAYOUT_PARAMS_DIR/path).stat().st_mtime>cached_time:
                            logger.info(f"LayoutParam cache is missing or out-of-date for {path}, regenerating.")
                            need_to_regenerate=True

        if need_to_regenerate:
            return loaded_yaml

    def regenerate_from_excel(self,yaml):
        self._yaml=yaml
        self._cat_tables: dict[(str,str),pd.DataFrame] ={}
        self._dut_to_catkey: dict[str,(str,str)] ={}
        self._tables_by_meas: dict[str,pd.DataFrame] ={}
        self._rows_by_mask: dict[str,list[str]] ={}

        for mask,paths_by_mask in yaml['layout_param_paths'].items():
            self._rows_by_mask[mask]=[]
            for path in paths_by_mask:
                logger.info(f"Reading {str(self.LAYOUT_PARAMS_DIR/path)}")
                with open(self.LAYOUT_PARAMS_DIR/path,'rb') as f:
                    xls=pd.ExcelFile(f,engine='openpyxl')
                    for sh in xls.book.sheetnames:
                        table=pd.read_excel(xls,sh).ffill()
                        if not ('rowname' in table.keys() and 'DUT' in table.keys()):
                            logger.debug(f"Ignoring {sh}")
                            continue
                        self._rows_by_mask[mask]+=list(table['rowname'].unique())
                        table=table.rename(columns={c:c.replace("\n"," ").strip() for c in table.columns})
                        table.drop(columns=[c for c in ['StructureName','origrowname'] if c in table],inplace=True)
                        table=table.rename(columns={k:v for k,v in yaml['replace_param_names'].items() if k in table.columns})
                        table['Structure']=(table['RowName']+"-DUT"+table['DUT'].astype(str).str.zfill(2)).astype('string')
                        table.set_index('Structure',inplace=True)
                        pad_cols=[col for col in table.columns if col[:4]=="PAD:"]
                        for c in pad_cols:
                            if table[c].dtype==np.dtype('O'):
                                table[c]=[int(p[3:]) for p in table[c]]
                        table=table.rename(columns={col:"PAD:"+col[4:].lower() for col in pad_cols})
                        table=table.convert_dtypes()
                        self._cat_tables[(mask,sh)]=table
                        for structure,r in table.iterrows():
                            self._dut_to_catkey[structure]=(mask,sh)

        logger.info("Collating by measurement group")
        by_meas_group=yaml['by_meas_group']
        for meas_key in by_meas_group:
            self._tables_by_meas[meas_key]=pd.DataFrame({})
            for mask in yaml['layout_param_paths']:
                if mask in by_meas_group[meas_key]:
                    for catregex in by_meas_group[meas_key][mask]:
                        for _mask,cat in self._cat_tables:
                            if _mask==mask and re.match(catregex,cat):
                                self._tables_by_meas[meas_key]=\
                                    self._tables_by_meas[meas_key].combine_first(self._cat_tables[(mask,cat)])
                if (afunc_dotpaths:=by_meas_group[meas_key].get('apply',None)):
                    for afunc_dotpath in afunc_dotpaths:
                        afunc=getattr(import_module(afunc_dotpath.split(":")[0]),
                                      afunc_dotpath.split(":")[1])
                        afunc(self._tables_by_meas[meas_key])
            if 'names' in by_meas_group[meas_key]:
                tab_to_rename=self._tables_by_meas[meas_key]
                del self._tables_by_meas[meas_key]
                for altname in by_meas_group[meas_key]['names']:
                    self._tables_by_meas[altname]=tab_to_rename

        logger.info("Saving cache")
        with open(self.CACHED_PATH,'wb') as f:
            pickle.dump(self,file=f)

    def __getitem__(self,item):
        if type(item) is str:
            res=self.get_params([item])
            return res.iloc[0]
        else:
            return self.get_params(item)

    def _get_correction_map(self,structures,mask):
        correction_map={}
        for structure in structures:
            if structure not in self._dut_to_catkey:
                supplied_row,dut=structure.split("-",maxsplit=1)
                corrected_row=self.search_partial_rowname(mask,supplied_row)
                if len(dut)!=5:
                    if dut.startswith("DUT"):
                        dut=int(dut[3:])
                    else:
                        dut=int(dut)
                    dut=f"DUT{dut:02d}"
                corrected_structure=corrected_row+'-'+dut
                correction_map[structure]=corrected_structure
        return correction_map


    def get_params(self,structures,drop_pads=True,for_measurement_group=None,allow_partial=False):
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
                rows.append(singular_row:=self._cat_tables[self._dut_to_catkey[structure]].loc[[structure]])
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
            if "#" in partial_rowname:
                partial_rowname=partial_rowname.replace('#','2' if mask in ['S31Cd2'] else '1')
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

    def validate_structures_in_meas_group(self,structures,meas_group):
        rejects=[structure for structure in structures if structure not in self._tables_by_meas[meas_group].index]
        assert len(rejects)==0, f"Structures {rejects} not in measurement parameter group '{meas_group}'"


    def merge_with_layout_params(self,meas_df,for_measurement_group,param_names=None, on_missing='error'):
        #structures_to_get=meas_df['Structure'].unique()
        #params=self.get_params(structures_to_get,allow_partial=False,for_measurement_group=for_measurement_group)
        params=self._tables_by_meas[for_measurement_group]
        if param_names:
            params=params[[pn for pn in param_names if pn in params.columns]].copy()
            for param in param_names:
                if param not in params.columns:
                    if on_missing=='error':
                        raise Exception(f"Missing parameter {param}")
                    elif on_missing=='NA':
                        params[param]=pd.NA
                    else:
                        raise Exception(f"Unrecognized value for on_missing={on_missing}")
        else:
            cols_to_drop=[c for c in ['RowName','DUT'] if c in meas_df]+[c for c in params if c.startswith("PAD:")]
            params=params.drop(columns=cols_to_drop)
        merged=pd.merge(left=meas_df,right=params,how='left',left_on=['Structure'],right_index=True,
                        suffixes=(None,'_param'))
        #import pdb; pdb.set_trace()
        return merged
