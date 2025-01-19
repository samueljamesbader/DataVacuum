import os
import re
from importlib import import_module
from pathlib import Path

from datavac.util.logging import logger, time_it
import pandas as pd
import numpy as np
import pickle
import traceback
import yaml


class LayoutParameters:
    _instance=None
    _db=None

    LAYOUT_PARAMS_DIR=Path(os.environ.get("DATAVACUUM_LAYOUT_PARAMS_DIR",Path.cwd()))

    # This singleton pattern gets really unwieldy, consider a factory_function instead...
    def __new__(cls,force_regenerate=False,database=None) -> "LayoutParameters":
        """ Database option is so Database can supply itself (partially built but with the Blob Store existing),
        since it needs LayoutParameters to exist too. """

        with time_it("DB setup for layout_params took",threshold_time=.1):
            if database is not None:
                cls._db=database
            if cls._db is None:
                from datavac.io.database import get_database;
                db=get_database(skip_establish=True);
                cls._db=db

        #import pdb; pdb.set_trace()

        # See if there's no singleton yet
        if cls._instance is None:

            # If there's no singleton, see if you need to regenerate (forced or cache-out-of-date)
            with time_it("Checking for need to regenerate layout_params took",threshold_time=.1):
                ytr=cls.yaml_to_regenerate(force_regenerate=force_regenerate)

            # If no need to regenerate, try unpickling and return the result
            if not ytr:
                try:
                    cls._instance=super().__new__(cls)
                    cls._instance=cls._db.get_obj('LayoutParams')
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
                cls._instance.regenerate_from_excel(*ytr)
        # If there is a singleton
        else:
            # Regenerate if required
            if force_regenerate:
                ytr=cls.yaml_to_regenerate(force_regenerate=force_regenerate)
                cls._instance.regenerate_from_excel(*ytr)

        # Return singleton
        return cls._instance

    @classmethod
    def yaml_to_regenerate(cls,force_regenerate=False):

        #yaml_path= Path(os.environ["DATAVACUUM_LAYOUT_PARAMS_YAML"])
        yaml_path = Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"layout_params.yaml"
        try: cached_time:float= cls._db.get_obj_date('LayoutParams').timestamp()
        except: cached_time:float= -np.inf

        need_to_regenerate=force_regenerate
        with open(yaml_path,'r') as f:
            loaded_yaml_str=f.read()
            loaded_yaml=yaml.safe_load(loaded_yaml_str)
        if yaml_path.stat().st_mtime>cached_time:
            logger.info("layout_params.yaml has more recent timestamp than cache...")
            try:
                last_used_yaml_str=cls._db.get_obj('layout_params.yaml')
                if last_used_yaml_str==loaded_yaml_str:
                    logger.info("...but is equivalent to the last-used one.")
                    need_to_regenerate=False
                else:
                    logger.info("...and is different from the last-used one.")
                    need_to_regenerate=True
            except:
                logger.info("...and previous one is unavailable.")
                need_to_regenerate=True
        else:
            try:
                last_used_yaml_str=cls._db.get_obj('layout_params.yaml')
            except:
                logger.info("layout_params.yaml has older timestamp than cache but can't get cached layout_params.yaml")
                need_to_regenerate=True
            else:
                if not (last_used_yaml_str==loaded_yaml_str):
                    raise Exception("layout_params.yaml is older than cached one and different.")
                else:
                    need_to_regenerate=False


        layout_param_paths=loaded_yaml['layout_param_paths']
        for mask,paths_by_mask in layout_param_paths.items():
            if need_to_regenerate: break
            for path in paths_by_mask:
                if not need_to_regenerate:
                    if (cls.LAYOUT_PARAMS_DIR/path).stat().st_mtime>cached_time:
                        logger.info(f"LayoutParam cache is missing or out-of-date for {path}, regenerating.")
                        need_to_regenerate=True; break

        if need_to_regenerate:
            return loaded_yaml,loaded_yaml_str

    def regenerate_from_excel(self,yaml,yaml_str):
        self._yaml=yaml
        self._cat_tables: dict[(str,str),pd.DataFrame] ={}
        self._dut_to_catkey: dict[(str,str),(str,str)] ={}
        self._tables_by_meas: dict[str,pd.DataFrame] ={}
        self._rows_by_mask: dict[str,list[str]] ={}

        for mask,paths_by_mask in yaml['layout_param_paths'].items():
            self._rows_by_mask[mask]=[]
            for path in paths_by_mask:
                logger.info(f"Reading {str(self.LAYOUT_PARAMS_DIR/path)}")
                with open(self.LAYOUT_PARAMS_DIR/path,'rb') as f:
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
                        for c in table.columns:
                            if table[c].dtype==np.dtype('O'):
                                logger.debug(f"Stringifying '{c}' in '{sh}'")
                                table[c]=table[c].astype('string')
                        self._cat_tables[(mask,sh)]=table
                        for structure,r in table.iterrows():
                            self._dut_to_catkey[(structure,mask)]=(mask,sh)

        logger.info("Collating by measurement group")
        by_meas_group=yaml['by_meas_group']
        for meas_key in by_meas_group:
            self._tables_by_meas[meas_key]=pd.DataFrame({})
            sensible_column_order=[]
            for mask in yaml['layout_param_paths']:
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

        logger.info("Saving cache")
        with self._db.engine.begin() as conn:
            self._db.store_obj('LayoutParams',self)
            self._db.store_obj('layout_params.yaml',yaml_str)

    def __getitem__(self,item):
        raise Exception("Didn't update this when I changed get_params to require mask")
        if type(item) is str:
            res=self.get_params([item])
            return res.iloc[0]
        else:
            return self.get_params(item)

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
