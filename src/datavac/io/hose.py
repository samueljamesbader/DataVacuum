import os
from pathlib import Path
from typing import Any

from sqlalchemy import text, select
import h5py

from datavac.io.measurement_table import MeasurementTable, MultiUniformMeasurementTable
from datavac.io.securepkl import SecurePkl
from datavac.util.logging import logger
import pandas as pd

CACHE_DIR=Path(os.environ['DATAVACUUM_CACHE_DIR'])

class Hose:
    def __init__(self, source_name:str, analysis_name:str):
        self.source_name=source_name
        self.analysis_name=analysis_name
        self._securepkl=SecurePkl()

    def get_factors(self,meas_group,parameter,**where):
        return self.get_where(meas_group,[parameter],distinct=True,**where)

    def get_data(self,meas_group,scalar_columns,raw_columns=[],**where):
        pass

    def _data_to_hdf5(self, lot: str, wafer: str, data_umts: dict[str,MultiUniformMeasurementTable]):
        logger.info(f"Saving HDF5")
        (CACHE_DIR/f"lots/{lot}").mkdir(parents=True,exist_ok=True)
        filename=CACHE_DIR/f"lots/{lot}/{lot}_{wafer}_{self.source_name}_{self.analysis_name}.hdf5"
        with h5py.File(filename,'w') as f:
            for k,v in data_umts.items():
                if hasattr(v,'to_hdf5_datasets'):
                    v.to_hdf5_datasets(f,k)
                elif hasattr(v,'to_hdf5_dataset'):
                    v.to_hdf5_dataset(f,k)
        logger.info(f"Done saving HDF5, dropping headers")

        for k in data_umts:
            if isinstance(data_umts[k],MeasurementTable):
                data_umts[k].drop_headers()

    def _irregular_data_to_pkl(self, lot:str, wafer: str, data_umts: dict[str,Any]):
        logger.info(f"Saving pickle")
        f=CACHE_DIR/f"lots/{lot}/{lot}_{wafer}_{self.source_name}_{self.analysis_name}.pkl"
        self._securepkl.secure_filedump({k:v for k,v in data_umts.items() if not isinstance(v,MeasurementTable)}, f)
        logger.info(f"Done saving pickle {str(f)}")

    def push_data(self, lot: str, wafer: str, data_umts: dict[str,MultiUniformMeasurementTable]):
        raise NotImplementedError

class DBHose(Hose):
    def __init__(self, source_name:str, analysis_name:str):
        super().__init__(source_name=source_name, analysis_name=analysis_name)
        from datavac.io.database import Database
        self._database=Database()

    def get_lots(self,meas_group):
        # Could replace with use of get_factors...
        table_name=f'{self.source_name}--{self.analysis_name}--{meas_group}--Summary'
        with self._database.get_engine().begin() as conn:
            res=conn.execute(text(f"SELECT DISTINCT \"LotWafer\" from \"{table_name}\""))
            return sorted(set([x[0].split("_")[0] for x in res]),reverse=True)

    def get_all_factors(self,meas_group,parameters,**where):

        assert os.environ["DATAVACUUM_DB_DRIVERNAME"]=='postgresql',\
            "Getting all factors at once is only supported by Postgresql"
        table_name=f'{self.source_name}--{self.analysis_name}--{meas_group}--Summary'
        tab=self._database.get_alchemy_table(table_name)
        dtypes=self._database.dtypes_from_alchemy_table(tab)
        from sqlalchemy.sql import union
        u=union(*[select(*[getattr(tab.c,p_) for p_ in parameters]).distinct(p) for p in parameters])
        u.compile()
        with self._database.get_engine().begin() as conn:
            conn.execute()




    def get_where(self,meas_group,columns,distinct=False,**where)->pd.DataFrame:
        #import pdb; pdb.set_trace()
        table_name=f'{self.source_name}--{self.analysis_name}--{meas_group}--Summary'
        tab=self._database.get_alchemy_table(table_name)
        dtypes=self._database.dtypes_from_alchemy_table(tab)
        #import pdb; pdb.set_trace()
        if columns=='*':
            columns=list(dtypes.keys())
            #raise NotImplementedError()
        try:
            stmt=select(*[getattr(tab.c,p) for p in columns])
        except:
            logger.debug(f"Problem creating query for {columns}, probably one of them isn't in the table.")
            raise
        if distinct:
            assert len(columns)==1
            stmt=stmt.distinct()
        for param,factor_list in where.items():
            if factor_list:# and len(factor_list):
                stmt=stmt.where(getattr(tab.c,param).in_(factor_list))
        import time
        def do_the_getting(conn):
            if distinct:
                return list(sorted([x[0] for x in conn.execute(stmt)]))
            else:
                res=list(conn.execute(stmt))
                if not len(res):
                    return pd.DataFrame({c:pd.Series([],dtype=dtypes[c]) for c in columns})
                return pd.DataFrame({c:pd.Series(coldata,dtype=dtypes[c]) for c,coldata in zip(columns,zip(*res))})
                #return pd.DataFrame(dict(zip(columns,zip(*res))))
        if True:
            start_time=time.time()
            with self._database.get_engine().begin() as conn:
                #print(f"Getting connection took {time.time()-start_time:.5g}s")
                return do_the_getting(conn)
        else:
            return do_the_getting(conn)


    def get_data(self,meas_group,scalar_columns="*",raw_columns=[],on_missing_column='error',**where):
        if len(raw_columns):
            assert (scalar_columns=='*') or ('LotWafer' in scalar_columns)
            assert 'MeasIndex' not in scalar_columns
            scalar_table=self.get_where(meas_group,('*' if scalar_columns=='*' else ['MeasIndex']+scalar_columns),**where)
            raw_data=[]
            for lotwafer,grp in scalar_table[['LotWafer','MeasIndex']].groupby('LotWafer'):
                lot,wafer=lotwafer.split("_")
                indices=list(grp['MeasIndex'].sort_values())
                if not len(indices): continue
                with h5py.File(CACHE_DIR/f"lots/{lot}/"\
                                 f"{lot}_{wafer}_{self.source_name}_{self.analysis_name}.hdf5",'r') as f:
                    mt=MultiUniformMeasurementTable.from_hdf5_datasets(
                                                                    f=f,meas_group=meas_group,headers=raw_columns,
                                                                    indices=indices,on_missing_column=on_missing_column)
                    df=mt._dataframe
                    df['LotWafer']=pd.Series([lotwafer]*len(df),dtype='string')
                    df['MeasIndex']=pd.Series(indices,dtype='Int64')
                    raw_data.append(df)

            ####import pdb; pdb.set_trace()
            if not len(raw_data):
                raw_data=[{}]
                for c in ['LotWafer','MeasIndex']+raw_columns: raw_data[0][c]=[]
                raw_data[0]=pd.DataFrame(raw_data[0]).astype({'LotWafer':'string','MeasIndex':'Int64'})
            #import pdb ;pdb.set_trace()
            return pd.merge(left=pd.concat(raw_data),right=scalar_table,
                            on=['LotWafer','MeasIndex']).drop(columns=['MeasIndex'])
        else:
            return self.get_where(meas_group,scalar_columns,**where)

    def push_data(self, lot: str, wafer: str, data_umts: dict[str,MultiUniformMeasurementTable],allow_delete_table=False):
        self._data_to_hdf5(lot=lot,wafer=wafer,data_umts=data_umts)
        self._irregular_data_to_pkl(lot=lot,wafer=wafer,data_umts=data_umts)
        logger.info("Uploading summary")
        self._database.upload_summary(self.source_name,lot,wafer,self.analysis_name,data_umts, allow_delete_table=allow_delete_table)
        logger.info("Done uploading summary")

class MeasTableHose(Hose):
    def __init__(self, source_name: str, analysis_name:str, mts:dict[str,MeasurementTable]):
        super().__init__(source_name,analysis_name)
        self._mts: dict[str,MeasurementTable] = mts

    def get_lots(self, meas_group):
        return list(sorted(set([lw.split("_")[0] for lw in
                                self.get_where(meas_group,['LotWafer'],distinct=True)])))

    def get_where(self,meas_group,columns,distinct=False,**where)->pd.DataFrame:
        df=self._mts[meas_group].scalar_table_with_layout_params()
        if len(where):
            df=df.query(" and ".join([f"`{k}` in {v}" for k,v in where.items() if v is not None and len(v)]),engine='python')
        if distinct:
            assert len(columns)==1
            return list(sorted(df[columns[0]].unique()))
        else:
            return df[columns]
    def get_data(self,meas_group,scalar_columns="*",raw_columns=[],on_missing_column='error',**where):
        df=self._mts[meas_group].full_table_with_layout_params()
        if len(where):
            df=df.query(" and ".join([f"`{k}` in {v}" for k,v in where.items() if v is not None and len(v)]),engine='python')
        if scalar_columns=='*':
            scalar_columns=list(self._mts[meas_group].scalar_table_with_layout_params().columns)
        columns=scalar_columns+raw_columns
        return df[columns]