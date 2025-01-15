import numpy as np
import pandas as pd

from datavac.io.layout_params import LayoutParameters
from datavac.util.tables import check_dtypes
from datavac.util.logging import logger
from datavac.measurements.measurement_type import MeasurementType

class MeasurementTable:

    def __init__(self,headers,meas_type,meas_group):
        self.meas_type: MeasurementType=meas_type
        self.meas_group: str=meas_group
        self.headers: list[str]=headers

    # TODO: These methods are probably no longer necessary now that project variable is gone
    def __getstate__(self):
        return (self.meas_type,self.meas_group,self.headers)
    def __setstate__(self, state):
        self.meas_type,self.meas_group,self.headers=state

    @property
    def _dataframe(self) -> pd.DataFrame:
        # Should be implemented by subclass
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self,item):
        return self._dataframe[item]

    def __contains__(self, item):
        return (item in self._dataframe)

    def drop(self,columns=None):
        raise NotImplementedError()

    def drop_headers(self):
        raise NotImplementedError()

    @property
    def scalar_table(self):
        return self._dataframe.drop(columns=self.headers)

    def scalar_table_with_layout_params(self, params=None, on_missing='error') -> pd.DataFrame:
        if self.meas_group:
            return LayoutParameters().merge_with_layout_params(
                self.scalar_table,self.meas_group,param_names=params,on_missing=on_missing)
        else:
            return self.scalar_table

    def full_table_with_layout_params(self, params=None, on_missing='error'):
        if self.meas_group:
            return LayoutParameters().merge_with_layout_params(
                self._dataframe,self.meas_group,param_names=params,on_missing=on_missing)
        else:
            return self._dataframe

    def analyze(self,*args,**kwargs):
        self.meas_type.analyze(self,*args,**kwargs)


class NonUniformMeasurementTable(MeasurementTable):
    def __init__(self,dataframe,meas_type,meas_group):
        super().__init__(headers=['RawData'],meas_type=meas_type,meas_group=meas_group)
        self._the_dataframe: pd.DataFrame=dataframe

    @property
    def _dataframe(self):
        return self._the_dataframe

    def __getstate__(self):
        return (super().__getstate__(),self._the_dataframe)
    def __setstate__(self, state):
        super_state,self._the_dataframe=state
        super().__setstate__(super_state)

    @staticmethod
    def from_read_data(read_dfs,meas_type,meas_group=None):
        assert len(read_dfs)==1, "I guess if I wanted I could merge these here..."
        return NonUniformMeasurementTable(dataframe=read_dfs[0],
            meas_type=meas_type, meas_group=meas_group)

    def __add__(self, other):
        assert self.headers==other.headers
        assert self.meas_type==other.meas_type, f"Meas type mismatch {self.meas_type} != {other.meas_type}"
        return NonUniformMeasurementTable(pd.concat([self._dataframe,other._dataframe],ignore_index=True),
                                self.meas_type,self.meas_group)


    def __setitem__(self,item,value):
        assert item not in self.headers
        self._the_dataframe.__setitem__(item,value)

    def drop(self,columns=None):
        assert all(c not in self.headers for c in columns)
        self._the_dataframe=self._the_dataframe.drop(columns=columns)

    def drop_headers(self):
        self._the_dataframe=self._the_dataframe.drop(columns=self.headers)
        self.headers=[]

    def assign_in_place(self,**kwargs):
        assert not any(c in self.headers for c in kwargs)
        self._the_dataframe=self._the_dataframe.assign(**kwargs)

class UniformMeasurementTable(MeasurementTable):

    def __init__(self,dataframe,headers,meas_type,meas_group,meas_length):
        super().__init__(headers=headers,meas_type=meas_type,meas_group=meas_group)
        assert isinstance(dataframe.index,pd.RangeIndex) and dataframe.index.start==0 and dataframe.index.step==1, \
            "Make sure the dataframe has a default index for UniformMeasurementTable"
        self._the_dataframe=dataframe.copy()
        self.meas_length=meas_length

        check_dtypes(self.scalar_table)

    @property
    def _dataframe(self):
        return self._the_dataframe

    def __add__(self, other):
        assert self.headers==other.headers, f"Header mismatch: {self.headers} != {other.headers}"
        assert self.meas_type==other.meas_type, f"Meas type mismatch {self.meas_type} != {other.meas_type}"
        if self.meas_length==other.meas_length:
            return UniformMeasurementTable(dataframe=pd.concat([self._dataframe,other._dataframe],ignore_index=True),
                                        headers=self.headers,meas_type=self.meas_type,meas_group=self.meas_group,meas_length=self.meas_length)
        else:
            logger.warning("Adding two UniformMeasurementTables of different meas_lengths"\
                           " to produce a MultiUniformMeasurementTable")
            return MultiUniformMeasurementTable([self,other])

    def __getitem__(self,item):
        if item in self.headers:
            return np.vstack(self._dataframe[item])
        else:
            return self._dataframe[item]

    def __setitem__(self,item,value):
        assert item not in self.headers
        if isinstance(value,pd.Series):
            assert isinstance(value.index,pd.RangeIndex) and value.index.start==0 and value.index.step==1, \
                "Make sure the value has a default index for setting to UniformMeasurementTable"
        #self._the_dataframe.__setitem__(item,value)
        self._the_dataframe[item]=value

    def drop(self,columns=None):
        assert all(c not in self.headers for c in columns)
        self._the_dataframe=self._the_dataframe.drop(columns=columns)

    def drop_headers(self):
        self._the_dataframe=self._the_dataframe.drop(columns=self.headers)
        self.headers=[]

    @staticmethod
    def from_read_data(read_dfs,meas_type,meas_group=None):
        assert len(read_dfs)==1, "I guess if I wanted I could merge these here..."
        read_df=read_dfs[0]

        meas_length=read_df['MeasLength'].iloc[0]
        assert all(n == meas_length for n in read_df['MeasLength']),\
            f"Multiple meas_length's: {read_df['MeasLength'].unique()}"

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
        return (meas_id_table, raw_meas_table, super_state, self.meas_length)

    def __setstate__(self, state):
        self._the_dataframe, raw_meas_table, super_state,  self.meas_length = state
        super().__setstate__(super_state)
        if raw_meas_table is not None:
            for header in raw_meas_table.columns:
                self._the_dataframe[header]=list(raw_meas_table[header].to_numpy().reshape(-1, self.meas_length))

    def assign_in_place(self,**kwargs):
        assert not any(c in self.headers for c in kwargs)
        self._the_dataframe=self._the_dataframe.assign(**kwargs)

    def to_hdf5_dataset(self,f,meas_group,i=0):
        for header in self.headers:
            f.create_dataset(f'{meas_group}/{i}/{header}',data=self[header])
        if i==0:
            self['MeasIndex']=list(self._dataframe.index)

    @staticmethod
    def from_hdf5_dataset(f, meas_group, i=0, headers=None, indices=None, on_missing_column='none'):
        headers=headers if headers else list(f[meas_group][i].keys())
        raw_data={}
        assert not ((indices is None or type(indices)==slice) and on_missing_column!='error'),\
            "won't know how many blank to put in... this can be figured out if needed by looking at others(?)"
        for c in headers:
            if c in f[meas_group][i]:
                raw_data[c]=list(f[meas_group][i][c][indices])
            else:
                if on_missing_column=='error':
                    raise Exception(f"Missing column {c} from {meas_group} in {f}."\
                                    f"  Columns present are {list(f[meas_group][i].keys())}")
                else:
                    raw_data[c]=[[] for i in indices]
        return UniformMeasurementTable(pd.DataFrame(raw_data),list(raw_data.keys()),None,
                                       meas_group,len(next(iter(raw_data.values()))))

    def defrag(self):
        self._the_dataframe=self._the_dataframe.copy()

class MultiUniformMeasurementTable(MeasurementTable):
    def __init__(self,umts: list[UniformMeasurementTable]):
        meas_type=umts[0].meas_type if len(umts) else MeasurementType
        meas_group=umts[0].meas_group if len(umts) else None
        super().__init__(headers=list(set([h for umt in umts for h in umt.headers])),
                         meas_type=meas_type, meas_group=meas_group)
        self._umts=umts
        #assert all(umt.headers==self.headers for umt in self._umts)
        assert all(umt.meas_type.__class__==self.meas_type.__class__ for umt in self._umts)
        assert all(umt.meas_group==self.meas_group for umt in self._umts)

    @staticmethod
    #@wraps(read_measurement_dataframe_from_excel,updated=('__annotations__',))
    def from_read_data(read_dfs,meas_type,meas_group=None):

        # Could potentially apply some logic to re-combine these into Uniform table iff they are compatible...
        umts=[]
        for read_df in read_dfs:
            if not len(read_df): continue
            meas_length=read_df['MeasLength'].iloc[0]
            assert all(n == meas_length for n in read_df['MeasLength']), \
                f"Multiple meas_length's: {read_df['MeasLength'].unique()}"

            headers=read_df['RawData'].iloc[0].keys()
            assert all(raw.keys() == headers for raw in read_df['RawData']),\
                f"Header mismatch: {[list(raw.keys()) for raw in read_df['RawData']]}"

            # Could just assign directly, but when there are many columns, that results in
            # highly-fragmented-data PerformanceWarnings from Pandas
            header_part=pd.DataFrame({col:[raw[col] for raw in read_df['RawData']] for col in headers})
            read_df=pd.concat([read_df,header_part],axis=1)

            umts.append(UniformMeasurementTable(
                dataframe=read_df.drop(columns=['RawData']),headers=list(headers),
                meas_length=meas_length,meas_type=meas_type,meas_group=meas_group))
        if not len(umts):
            raise Exception(f"No data for {meas_group}")
        return MultiUniformMeasurementTable(umts=umts)

    def __add__(self, other):
        return MultiUniformMeasurementTable(self._umts+other._umts)

    def __getstate__(self):
        return self._umts

    def __setstate__(self,state):
        umts=state
        self.__init__(umts)

    def analyze(self,*args,**kwargs):
        for umt in self._umts:
            umt.analyze(*args,**kwargs)
        umt_headers=[umt.headers for umt in self._umts]
        self.headers=umt_headers[0]
        try:
            assert all(h==self.headers for h in umt_headers), "Header mismatch after analysis"
        except Exception as e:
            old_headers=self.headers
            self.headers=list(set([h for umt in self._umts for h in umt.headers]))
            logger.warning(str(e))
            logger.warning(f"WARNING: Expanding header collection from {old_headers} to {self.headers}")


    def __len__(self):
        return sum([len(umt) for umt in self._umts])

    def __contains__(self, item):
        return (item in self._umts[0])

    @property
    def _dataframe(self):
        return pd.concat([umt._dataframe.assign(rawgroup=i) for i,umt in enumerate(self._umts)],ignore_index=True)

    @property
    def scalar_table(self):
        return pd.concat([umt._dataframe.drop(columns=umt.headers) for umt in self._umts],ignore_index=True)

    def __getitem__(self,item):
        if item in self.headers:
            return np.vstack(self._dataframe[item])
        else:
            return self._dataframe[item]

    def __setitem__(self,item,value):
        assert item not in self.headers
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

    def drop(self,columns=None):
        assert all(c not in self.headers for c in columns)
        for umt in self._umts:
            umt.drop(columns=columns)

    def drop_headers(self):
        for umt in self._umts:
            umt.drop_headers()
        self.headers=[]

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

    def to_hdf5_datasets(self,f,meas_group):
        for i,umt in enumerate(self._umts):
            umt.to_hdf5_dataset(f=f,meas_group=meas_group,i=i)
        self['MeasIndex']=list(self._dataframe.index)

    @staticmethod
    def from_hdf5_datasets(f, meas_group, headers, indices=None, on_missing_column='none'):
        umts=[]
        sub_umt_nos=[str(x) for x in sorted([int(x) for x in f[meas_group].keys()])]
        assert len(sub_umt_nos)==len(set(sub_umt_nos))

        umt_start_index=0
        for k in sub_umt_nos:
            umt_len=next(iter(f[meas_group][k].values())).len()
            if indices:
                sub_indices=[i-umt_start_index for i in indices if umt_start_index <= i < umt_start_index+umt_len]
                umt_start_index+=umt_len
                if not len(sub_indices): continue
            else:
                sub_indices=slice(None)
            umts.append(UniformMeasurementTable.from_hdf5_dataset(f=f,
                                                      meas_group=meas_group,i=k,headers=headers,
                                                      indices=sub_indices,on_missing_column=on_missing_column))
        return MultiUniformMeasurementTable(umts=umts)

    def defrag(self):
        for umt in self._umts:
            umt.defrag()