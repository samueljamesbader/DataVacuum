from typing import Any

import numpy as np
import pandas as pd

from datavac.io.layout_params import get_layout_params
from datavac.util.tables import check_dtypes
from datavac.util.logging import logger
from datavac.measurements.measurement_type import MeasurementType
from datavac.util.util import only


class MeasurementTable:

    def __init__(self, meas_type: MeasurementType, meas_group: str):
        self.meas_type: MeasurementType = meas_type
        self.meas_group: str = meas_group

    def __add__(self, other):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def drop(self, columns: list[str]):
        raise NotImplementedError()

    def get_stacked_sweeps(self):
        raise NotImplementedError()

    @property
    def scalar_table(self):
        raise NotImplementedError()

class DataFrameBackedMeasurementTable(MeasurementTable):

    def __init__(self, dataframe: pd.DataFrame, meas_type: MeasurementType,
                 meas_group: str, non_scalar_columns: list[str] = ()):
        super().__init__(meas_type,meas_group)
        self._the_dataframe: pd.DataFrame = dataframe.copy()
        self._non_scalar_columns: list[str] = list(non_scalar_columns)

    @property
    def _dataframe(self):
        return self._the_dataframe

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self,item):
        return self._dataframe[item]

    def __setitem__(self,item,value):
        assert item not in self._non_scalar_columns
        self._the_dataframe.__setitem__(item,value)

    def __contains__(self, item):
        return item in self._dataframe.columns

    def assign_in_place(self,**kwargs):
        assert not any(c in self._non_scalar_columns for c in kwargs)
        self._the_dataframe=self._the_dataframe.assign(**kwargs)

    @property
    def scalar_table(self):
        return self._dataframe.drop(columns=[h for h in self._non_scalar_columns
                                             if h in self._dataframe.columns])

    def scalar_table_with_layout_params(self, params=None, on_missing='error') -> pd.DataFrame:
        if self.meas_group:
            return get_layout_params().merge_with_layout_params(
                self.scalar_table,self.meas_group,param_names=params,on_missing=on_missing)
        else:
            return self.scalar_table

    def full_table_with_layout_params(self, params=None, on_missing='error'):
        if self.meas_group:
            return get_layout_params().merge_with_layout_params(
                self._dataframe,self.meas_group,param_names=params,on_missing=on_missing)
        else:
            return self._dataframe

    def drop(self,columns: list[str]):
        assert all(c not in self._non_scalar_columns for c in columns)
        self._the_dataframe=self._the_dataframe.drop(columns=columns)

    def defrag(self):
        self._the_dataframe=self._the_dataframe.copy()


class NonUniformMeasurementTable(DataFrameBackedMeasurementTable):
    def __init__(self, dataframe, meas_type, meas_group):
        super().__init__(dataframe=dataframe,non_scalar_columns=['RawData'],
                         meas_type=meas_type,meas_group=meas_group)

    @staticmethod
    def from_read_data(read_dfs,meas_type,meas_group=None):
        assert len(read_dfs)==1, "I guess if I wanted I could merge these here..."
        return NonUniformMeasurementTable(dataframe=read_dfs[0],
            meas_type=meas_type, meas_group=meas_group)

    def __add__(self, other: 'NonUniformMeasurementTable'):
        assert isinstance(other, NonUniformMeasurementTable),\
            "NonUniformMeasurementTable can only be added to other NonUniformMeasurementTable"
        assert self.meas_type==other.meas_type,\
            f"Meas type mismatch {self.meas_type} != {other.meas_type}"
        return NonUniformMeasurementTable(
            pd.concat([self._dataframe,other._dataframe],ignore_index=True),
            self.meas_type,self.meas_group)

class UniformMeasurementTable(DataFrameBackedMeasurementTable):

    def __init__(self, dataframe: pd.DataFrame, headers: list[str],
                 meas_type: MeasurementType, meas_group: str, meas_length: int):
        super().__init__(dataframe=dataframe, non_scalar_columns=headers,
                         meas_type=meas_type,meas_group=meas_group)
        assert isinstance(dataframe.index,pd.RangeIndex)\
               and dataframe.index.start==0 and dataframe.index.step==1, \
            "Make sure the dataframe has a default index for UniformMeasurementTable"
        self.meas_length=meas_length
        check_dtypes(self.scalar_table)

    @property
    def _dataframe(self):
        return self._the_dataframe

    @property
    def headers(self):
        return self._non_scalar_columns

    def __add__(self, other: 'UniformMeasurementTable'):
        assert self.headers==other.headers,\
            f"Header mismatch: {self.headers} != {other.headers}"
        assert self.meas_type==other.meas_type,\
            f"Meas type mismatch {self.meas_type} != {other.meas_type}"
        if self.meas_length==other.meas_length:
            return UniformMeasurementTable(
                dataframe=pd.concat([self._dataframe,other._dataframe],ignore_index=True),
                headers=self.headers,meas_type=self.meas_type,meas_group=self.meas_group,meas_length=self.meas_length)
        else:
            logger.warning("Adding two UniformMeasurementTables of different meas_lengths" 
                           " to produce a MultiUniformMeasurementTable")
            return MultiUniformMeasurementTable([self,other])

    def __getitem__(self, item: str):
        if item in self.headers:
            return np.vstack(self._dataframe[item])
        else:
            return self._dataframe[item]

    def __setitem__(self, item: str, value: Any):
        assert item not in self.headers
        if isinstance(value,pd.Series):
            assert isinstance(value.index,pd.RangeIndex) and value.index.start==0 and value.index.step==1, \
                "Make sure the value has a default index for setting to UniformMeasurementTable"
        #self._the_dataframe.__setitem__(item,value)
        self._the_dataframe[item]=value

    def drop(self, columns: list[str]):
        assert all(c not in self.headers for c in columns)
        super().drop(columns)

    def drop_headers(self):
        self._the_dataframe=self._the_dataframe.drop(columns=self.headers)
        self._non_scalar_columns=[]

    def analyze(self,*args,**kwargs):
        self.meas_type.analyze(self,*args,**kwargs)

    @staticmethod
    def from_read_data(read_dfs,meas_type,meas_group=None):
        assert len(read_dfs)==1, "I guess if I wanted I could merge these here..."
        read_df=read_dfs[0]

        if 'MeasLength' not in read_df.columns:
            #logger.warning("No MeasLength column in read data, will try to recreate")
            read_df=read_df.copy()
            read_df['MeasLength']=read_df['RawData'].apply(lambda rd:\
                                       only(set(len(rdv) for rdv in rd.values())))

        meas_length=read_df['MeasLength'].iloc[0]
        if not all(n == meas_length for n in read_df['MeasLength']):
            logger.warning('Multiple meas lengths... here\'s some one example location for each\n'+
                           str(read_df[['DieX','DieY','Site','MeasLength']].drop_duplicates(['MeasLength'])))
            raise Exception(f"Multiple meas_length's: {read_df['MeasLength'].unique()}")

        headers=read_df['RawData'].iloc[0].keys()
        assert all(raw.keys() == headers for raw in read_df['RawData'])

        for col in headers:
            read_df[col]=[raw[col] for raw in read_df['RawData']]

        return UniformMeasurementTable(
            dataframe=read_df.drop(columns=['RawData']),headers=list(headers),
            meas_length=meas_length,meas_type=meas_type,meas_group=meas_group)

    def __getstate__(self):
        super_state=super().__getstate__()
        meas_id_table=self._dataframe.drop(columns=self.headers)
        if len(self.headers):
            raw_meas_table=self._dataframe[self.headers].explode(column=self.headers, ignore_index=True)
            for col in self.headers:
                raw_meas_table[col]=np.array(raw_meas_table[col],dtype=self.meas_type.get_preferred_dtype(col))
        else:
            raw_meas_table=None
        return (meas_id_table, raw_meas_table,
                self.meas_type, self.meas_group, self.headers, self.meas_length)

    def __setstate__(self, state):
        self._the_dataframe, raw_meas_table,\
            self.meas_type, self.meas_group, self._non_scalar_columns,  self.meas_length = state
        if raw_meas_table is not None:
            for header in raw_meas_table.columns:
                self._the_dataframe[header]=list(raw_meas_table[header].to_numpy().reshape(-1, self.meas_length))

    def assign_in_place(self,**kwargs):
        assert not any(c in self.headers for c in kwargs)
        self._the_dataframe=self._the_dataframe.assign(**kwargs)

    def get_stacked_sweeps(self):
        return self._the_dataframe[self.headers]\
                        .stack().reset_index()\
                        .rename(columns={'level_0':'measid','level_1':'header',0:'sweep'})\

#def to_hdf5_dataset(self,f,meas_group,i=0):
    #    raise NotImplementedError("HDF5 methods have not been tested/maintained")
    #    for header in self.headers:
    #        f.create_dataset(f'{meas_group}/{i}/{header}',data=self[header])
    #    if i==0:
    #        self['MeasIndex']=list(self._dataframe.index)

    #@staticmethod
    #def from_hdf5_dataset(f, meas_group, i=0, headers=None, indices=None, on_missing_column='none'):
    #    raise NotImplementedError("HDF5 methods have not been tested/maintained")
    #    headers=headers if headers else list(f[meas_group][i].keys())
    #    raw_data={}
    #    assert not ((indices is None or type(indices)==slice) and on_missing_column!='error'),\
    #        "won't know how many blank to put in... this can be figured out if needed by looking at #others(?)"
    #    for c in headers:
    #        if c in f[meas_group][i]:
    #            raw_data[c]=list(f[meas_group][i][c][indices])
    #        else:
    #            if on_missing_column=='error':
    #                raise Exception(f"Missing column {c} from {meas_group} in {f}."\
    #                                f"  Columns present are {list(f[meas_group][i].keys())}")
    #            else:
    #                raw_data[c]=[[] for i in indices]
    #    return UniformMeasurementTable(pd.DataFrame(raw_data),list(raw_data.keys()),None,
    #                                   meas_group,len(next(iter(raw_data.values()))))

class MultiUniformMeasurementTable(MeasurementTable):
    def __init__(self, umts: list[UniformMeasurementTable]):
        meas_type=umts[0].meas_type if len(umts) else MeasurementType
        meas_group=umts[0].meas_group if len(umts) else None
        super().__init__(meas_type=meas_type, meas_group=meas_group)

        self._umts=umts

        # TODO: Refine this requirement
        assert all(umt.meas_type.__class__==self.meas_type.__class__ for umt in self._umts)
        assert all(umt.meas_group==self.meas_group for umt in self._umts)

    @staticmethod
    def from_read_data(read_dfs,meas_type,meas_group=None):

        # Could potentially apply some logic to re-combine these into Uniform table iff they are compatible...
        umts=[]
        for read_df in read_dfs:
            if not len(read_df): continue

            # TODO: remove this
            headers=read_df['RawData'].iloc[0].keys()
            if not all(raw.keys() == headers for raw in read_df['RawData']):
                logger.debug(f"Header mismatch: {[list(raw.keys()) for raw in read_df['RawData']]}")

            # Could just assign directly, but when there are many columns, that results in
            # highly-fragmented-data PerformanceWarnings from Pandas
            header_part=pd.DataFrame({col:[raw[col] for raw in read_df['RawData']] for col in headers})
            read_df=pd.concat([read_df,header_part],axis=1)

            umts.append(UniformMeasurementTable.from_read_data(
                read_dfs=[read_df],meas_type=meas_type,meas_group=meas_group))
        if not len(umts):
            raise Exception(f"No data for {meas_group}")
        return MultiUniformMeasurementTable(umts=umts)

    def __add__(self, other):
        return MultiUniformMeasurementTable(self._umts+other._umts)

    def analyze(self,*args,**kwargs):
        for umt in self._umts:
            umt.analyze(*args,**kwargs)

    def __len__(self):
        return sum([len(umt) for umt in self._umts])

    def __contains__(self, item):
        return item in self._umts[0]

    @property
    def _dataframe(self):
        return pd.concat([umt._dataframe.assign(rawgroup=i)
                          for i,umt in enumerate(self._umts)],ignore_index=True)

    @property
    def scalar_table(self):
        return pd.concat([umt._dataframe.drop(columns=umt.headers).assign(rawgroup=i)
                          for i,umt in enumerate(self._umts)],ignore_index=True)

    def __getitem__(self,item):
        if item in (h for umt in self._umts for h in umt.headers):
            assert len(set(umt.meas_length for umt in self._umts))==1
            return np.vstack(self._dataframe[item])
        else:
            return self._dataframe[item]

    def __setitem__(self,item,value):
        istart=0
        for umt in self._umts:
            istop=istart+len(umt)
            if isinstance(value,pd.Series) or isinstance(value,list) or isinstance(value,np.ndarray):
                subvalue=value[istart:istop]
                if isinstance(subvalue,pd.Series):
                    subvalue=subvalue.reset_index(drop=True)
            else:
                subvalue=value
            umt.__setitem__(item,subvalue)
            istart=istop

    def drop(self,columns:list[str]):
        for umt in self._umts:
            umt.drop(columns)

    def drop_headers(self):
        for umt in self._umts:
            umt.drop_headers()

    def assign_in_place(self,**kwargs):
        #if 'pol' in kwargs:
        #    import pdb; pdb.set_trace()
        evaled={}
        for c, val in kwargs.items():
            if callable(c):
                evaled[c]=[val for umt in self._umts]
            elif isinstance(c,pd.Series) or isinstance(c,list) or isinstance(c,np.ndarray):
                evaled[c]=[]
                istart=0
                for umt in self._umts:
                    istop=istart+len(umt)
                    subvalue=val[istart:istop]
                    if isinstance(c,pd.Series):
                        subvalue=subvalue.reset_index(drop=True)
                    evaled[c].append(subvalue)
                    istart=istop
            else:
                evaled[c]=[val for umt in self._umts]

        for i,umt in enumerate(self._umts):
            umt.assign_in_place(**{k:v[i] for k,v in evaled.items()})

    #def to_hdf5_datasets(self,f,meas_group):
    #    raise NotImplementedError("HDF5 methods have not been tested/maintained")
    #    for i,umt in enumerate(self._umts):
    #        umt.to_hdf5_dataset(f=f,meas_group=meas_group,i=i)
    #    self['MeasIndex']=list(self._dataframe.index)

    #@staticmethod
    #def from_hdf5_datasets(f, meas_group, headers, indices=None, on_missing_column='none'):
    #    raise NotImplementedError("HDF5 methods have not been tested/maintained")
    #    umts=[hh]
    #    sub_umt_nos=[str(x) for x in sorted([int(x) for x in f[meas_group].keys()])]
    #    assert len(sub_umt_nos)==len(set(sub_umt_nos))

    #    umt_start_index=0
    #    for k in sub_umt_nos:
    #        umt_len=next(iter(f[meas_group][k].values())).len()
    #        if indices:
    #            sub_indices=[i-umt_start_index for i in indices if umt_start_index <= i < #umt_start_index+umt_len]
    #            umt_start_index+=umt_len
    #            if not len(sub_indices): continue
    #        else:
    #            sub_indices=slice(None)
    #        umts.append(UniformMeasurementTable.from_hdf5_dataset(f=f,
    #                                                  meas_group=meas_group,i=k,headers=headers,
    #                                                  indices=sub_indices,on_missing_column=on_missing_column))
    #    return MultiUniformMeasurementTable(umts=umts)

    def defrag(self):
        for umt in self._umts:
            umt.defrag()

    def get_stacked_sweeps(self):
        subs=[]
        prev_meas_id=0
        for umt in self._umts:
            subs.append((ss:=umt.get_stacked_sweeps()))
            ss['measid']+=prev_meas_id
            prev_meas_id+=len(umt)
        return pd.concat(subs,ignore_index=True)
