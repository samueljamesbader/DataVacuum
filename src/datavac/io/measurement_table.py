from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union


from datavac.util.tables import check_dtypes
from datavac.util.dvlogging import logger
from datavac.measurements.measurement_type import MeasurementType
from datavac.util.util import only

if TYPE_CHECKING:
    from datavac.measurements.measurement_group import MeasurementGroup
    import numpy as np
    import pandas as pd

class MeasurementTable:

    def __init__(self, meas_group: MeasurementGroup):
        self.meas_group: MeasurementGroup = meas_group

    def __add__(self, other):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self,item):
        raise NotImplementedError()

    def drop(self, columns: list[str]):
        raise NotImplementedError()

    def get_stacked_sweeps(self,only_extr:bool=False) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def scalar_table(self):
        raise NotImplementedError()

    def scalar_table_with_layout_params(self, params=None, on_missing='error') -> pd.DataFrame:
        if self.meas_group:
            from datavac.config.layout_params import LP
            return LP().merge_with_layout_params(
                self.scalar_table,self.meas_group,param_names=params,on_missing=on_missing)
        else:
            return self.scalar_table


class DataFrameBackedMeasurementTable(MeasurementTable):

    def __init__(self, dataframe: pd.DataFrame, 
                 meas_group: MeasurementGroup, non_scalar_columns: Sequence[str] = ()):
        super().__init__(meas_group=meas_group)
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

    def full_table_with_layout_params(self, params=None, on_missing='error'):
        if self.meas_group:
            from datavac.io.layout_params import get_layout_params
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
    def __init__(self, dataframe: pd.DataFrame, meas_group: MeasurementGroup):
        super().__init__(dataframe=dataframe,non_scalar_columns=['RawData'],
                         meas_group=meas_group)

    @staticmethod
    def from_read_data(read_dfs: Sequence[pd.DataFrame], meas_group: MeasurementGroup):
        assert len(read_dfs)==1, "I guess if I wanted I could merge these here..."
        return NonUniformMeasurementTable(dataframe=read_dfs[0],
            meas_group=meas_group)

    def __add__(self, other: 'NonUniformMeasurementTable'):
        assert isinstance(other, NonUniformMeasurementTable),\
            "NonUniformMeasurementTable can only be added to other NonUniformMeasurementTable"
        assert self.meas_group==other.meas_group,\
            f"Meas group mismatch {self.meas_group} != {other.meas_group}"
        import pandas as pd
        return NonUniformMeasurementTable(
            pd.concat([self._dataframe,other._dataframe],ignore_index=True),
            meas_group=self.meas_group)

class UMTMUMT_H:
    #def __getattr__(self, item: str) -> np.ndarray: return self[item]
    #def __setattr__(self, item: str, value: Any): self.__setitem__(item, value)
    def __getitem__(self, item: str) -> np.ndarray: raise NotImplementedError()
    def __setitem__(self, item: str, value: Any): raise NotImplementedError()
class UMTMUMT_S:
    #def __getattr__(self, item: str) -> pd.Series:
    #    import pdb; pdb.set_trace()
    #    return self[item]
    #def __setattr__(self, item: str, value: Any): self.__setitem__(item, value)
    def __getitem__(self, item: str) -> pd.Series: raise NotImplementedError()
    def __setitem__(self, item: str, value: Any): raise NotImplementedError()

class UniformMeasurementTable(DataFrameBackedMeasurementTable):

    def __init__(self, dataframe: pd.DataFrame, headers: list[str],
                 meas_group: MeasurementGroup, meas_length: int,
                 extr_headers: Sequence[str]=()):
        super().__init__(dataframe=dataframe, non_scalar_columns=headers,
                         meas_group=meas_group)
        import pandas as pd
        assert isinstance(dataframe.index,pd.RangeIndex)\
               and dataframe.index.start==0 and dataframe.index.step==1, \
            "Make sure the dataframe has a default index for UniformMeasurementTable"
        self.meas_length=meas_length
        self.extr_headers=extr_headers
        check_dtypes(self.scalar_table)

    @property
    def _dataframe(self):
        return self._the_dataframe

    @property
    def headers(self):
        return self._non_scalar_columns
    
    def restricted_to_columns(self, columns: list[str]) -> 'UniformMeasurementTable':
        """Returns a new UniformMeasurementTable with only the specified (scalar) columns and no header columns."""
        for c in columns: assert c not in self.headers, \
            f"Cannot restrict to {columns} as it contains non-scalar columns {self.headers}"
        new_df = self._dataframe[columns].copy()
        return self.__class__(dataframe=new_df, headers=[],
                              meas_group=self.meas_group,
                              meas_length=self.meas_length)

    def __add__(self, other: 'UniformMeasurementTable'):
        #assert self.headers==other.headers,\
        #    f"Header mismatch: {self.headers} != {other.headers}"
        #assert self.meas_group==other.meas_group,\
        #    f"Meas group mismatch {self.meas_group} != {other.meas_group}"
        #if self.meas_length==other.meas_length:
        #    return UniformMeasurementTable(
        #        dataframe=pd.concat([self._dataframe,other._dataframe],ignore_index=True),
        #        headers=self.headers,meas_group=self.meas_group,meas_length=self.meas_length)
        #else:
        #    logger.warning("Adding two UniformMeasurementTables of different meas_lengths" 
        #                   " to produce a MultiUniformMeasurementTable")
        if True:
            return MultiUniformMeasurementTable([self,other])
    
    @property
    def h(self) -> UMTMUMT_H:
        class UMT_H(UMTMUMT_H):
            def __init__(self, umt: UniformMeasurementTable): self._umt = umt
            def __getitem__(self, item: str) -> np.ndarray:
                assert item in self._umt.headers
                import numpy as np
                return np.vstack(self._umt._dataframe[item]) # type: ignore
            def __setitem__(self, item: str, value: Any):
                raise NotImplementedError("Cannot set headers in UniformMeasurementTable")
        return UMT_H(self)
    @property
    def s(self) -> UMTMUMT_S:
        class UMT_S(UMTMUMT_S):
            def __init__(self, umt: UniformMeasurementTable): self._umt = umt
            def __getitem__(self, item: str) -> pd.Series:
                assert item not in self._umt.headers
                return self._umt._dataframe[item] # type: ignore
            def __setitem__(self, item: str, value: Any):
                assert item not in self._umt.headers
                self._umt._the_dataframe[item] = value
        return UMT_S(self)
            

    def __getitem__(self, item: str)->Union[np.ndarray, pd.Series]:
        #logger.debug(f"Accessing column {item} in UniformMeasurementTable as UMT[col] is deprecated, use UMT.s[col] or UMT.h[col] instead.")
        if item in self.headers:
            import numpy as np
            return np.vstack(self._dataframe[item]) # type: ignore
        else:
            try:
                return self._dataframe[item]
            except KeyError:
                logger.debug(f"Column {item} not found, did you mean {self._dataframe.columns.tolist()}?")
                raise

    def __setitem__(self, item: str, value: Any):
        #logger.debug(f"Setting column {item} in UniformMeasurementTable via UMT[col] is deprecated, use UMT.s[col] or UMT.h[col] instead.")
        assert item not in self.headers
        import pandas as pd
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
        raise Exception("This functionality has been moved to MeasurementGroup.extract_by_mumt/umt()")
        self.meas_group.extract_by_umt(self,*args,**kwargs)

    @staticmethod
    def from_read_data(read_dfs:Sequence[pd.DataFrame],meas_group:MeasurementGroup):
        assert len(read_dfs)==1, "I guess if I wanted I could merge these here..."
        read_df=read_dfs[0]

        if 'MeasLength' not in read_df.columns:
            #logger.warning("No MeasLength column in read data, will try to recreate")
            read_df=read_df.copy()
            if 'RawData' in read_df.columns:
                read_df['MeasLength']=read_df['RawData'].apply(lambda rd:\
                                           only(set(len(rdv) for rdv in rd.values())))
            else:
                read_df['MeasLength']=0

        meas_length=read_df['MeasLength'].iloc[0]
        if not all(n == meas_length for n in read_df['MeasLength']):
            logger.warning('Multiple meas lengths... here\'s some one example location for each\n'+
                           str(read_df[['DieX','DieY','Site','MeasLength']].drop_duplicates(['MeasLength'])))
            raise Exception(f"Multiple meas_length's: {read_df['MeasLength'].unique()}")

        if 'RawData' in read_df.columns:
            headers=read_df['RawData'].iloc[0].keys()
            if not all(raw.keys() == headers for raw in read_df['RawData']):
                logger.debug(f"Header mismatch: {[list(raw.keys()) for raw in read_df['RawData']]}")

            # Could just assign directly instead of making a separate dataframe,
            # but when there are many header columns to add,
            # that results in highly-fragmented-data PerformanceWarnings from Pandas
            import pandas as pd
            header_part=pd.DataFrame({col:[raw[col] for raw in read_df['RawData']] for col in headers})
            read_df=pd.concat([read_df,header_part],axis=1).drop(columns=['RawData'])
        else: headers=[]

        return UniformMeasurementTable(
            dataframe=read_df,headers=list(headers),
            meas_length=meas_length,meas_group=meas_group)

    def __getstate__(self):
        raise NotImplementedError("Pickling hasn't ben tested/maintained for UniformMeasurementTable")
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
        raise NotImplementedError("Pickling hasn't ben tested/maintained for UniformMeasurementTable")
        self._the_dataframe, raw_meas_table,\
            self.meas_type, self.meas_group, self._non_scalar_columns,  self.meas_length = state
        if raw_meas_table is not None:
            for header in raw_meas_table.columns:
                self._the_dataframe[header]=list(raw_meas_table[header].to_numpy().reshape(-1, self.meas_length))

    def assign_in_place(self,**kwargs):
        assert not any(c in self.headers for c in kwargs)
        self._the_dataframe=self._the_dataframe.assign(**kwargs)
    
    def add_extr_headers(self,**new_headers):
        assert all(h not in self.headers for h in new_headers),\
                f"Can't add extr headers {new_headers.keys()} as they already exist in {self.headers}"
        import pandas as pd
        self._the_dataframe=pd.DataFrame(
            dict(**self._the_dataframe.to_dict('series'),**new_headers)) # type: ignore
        self.extr_headers=list(self.extr_headers)+list(new_headers.keys())
        self._non_scalar_columns.extend(new_headers.keys())

    def get_stacked_sweeps(self,only_extr:bool=False) -> pd.DataFrame:
        import pandas as pd
        headers= list(self.extr_headers if only_extr else self.headers)
        if not len(headers):
            return pd.DataFrame(columns=['measid','header','sweep','israw'])
        df= self._the_dataframe[headers]\
                .stack().reset_index()\
                .rename(columns={'level_0':'measid','level_1':'header',0:'sweep'})
        israwmap = {h:not(h in self.extr_headers) for h in self.headers}
        df['israw']=df['header'].map(israwmap)
        return df

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
        meas_group=umts[0].meas_group if len(umts) else None
        super().__init__(meas_group=meas_group) # type: ignore # allowing None meas_group for empty umts

        self._umts=umts

        # TODO: Refine this requirement
        assert all(umt.meas_group==self.meas_group for umt in self._umts)

    @staticmethod
    def from_read_data(read_dfs,meas_group: MeasurementGroup):

        # Could potentially apply some logic to re-combine these into Uniform table iff they are compatible...
        umts=[]
        for read_df in read_dfs:
            if not len(read_df): continue

            umts.append(UniformMeasurementTable.from_read_data(
                read_dfs=[read_df],meas_group=meas_group))
        if not len(umts):
            raise Exception(f"No data for {meas_group}")
        return MultiUniformMeasurementTable(umts=umts)

    def __add__(self, other):
        return MultiUniformMeasurementTable(self._umts+other._umts)

    def analyze(self,*args,**kwargs):
        raise Exception("This functionality has been moved to MeasurementGroup.extract_by_mumt/umt()")
        for umt in self._umts:
            umt.analyze(*args,**kwargs)

    def __len__(self):
        return sum([len(umt) for umt in self._umts])

    def __contains__(self, item):
        return item in self._umts[0]
    
    def restricted_to_columns(self, columns: list[str]) -> 'MultiUniformMeasurementTable':
        """Returns a new MultiUniformMeasurementTable with only the specified (scalar) columns."""
        for c in columns: assert c not in (h for umt in self._umts for h in umt.headers), \
            f"Cannot restrict to {columns} as it contains non-scalar columns {[h for umt in self._umts for h in umt.headers]}"
        new_umts = [umt.restricted_to_columns(columns) for umt in self._umts]
        return MultiUniformMeasurementTable(umts=new_umts)

    @property
    def _dataframe(self):
        import pandas as pd
        return pd.concat([umt._dataframe.assign(rawgroup=i)
                          for i,umt in enumerate(self._umts)],ignore_index=True)

    @property
    def scalar_table(self):
        import pandas as pd
        return pd.concat([umt._dataframe.drop(columns=umt.headers).assign(rawgroup=i)
                          for i,umt in enumerate(self._umts)],ignore_index=True)

    @property
    def h(self) -> UMTMUMT_H:
        class MUMT_H(UMTMUMT_H):
            def __init__(self, mumt: MultiUniformMeasurementTable): self._mumt = mumt
            def __getitem__(self, item: str) -> np.ndarray:
                assert item in (h for umt in self._mumt._umts for h in umt.headers)
                assert len(set(umt.meas_length for umt in self._mumt._umts))==1
                import numpy as np
                return np.vstack(self._mumt._dataframe[item]) # type: ignore
            def __setitem__(self, item: str, value: Any):
                raise NotImplementedError("Cannot set headers in MultiUniformMeasurementTable")
        return MUMT_H(self)
    @property
    def s(self) -> UMTMUMT_S:
        class MUMT_S(UMTMUMT_S):
            def __init__(self, mumt: MultiUniformMeasurementTable): self._mumt = mumt
            def __getitem__(self, item: str) -> pd.Series:
                assert item not in (h for umt in self._mumt._umts for h in umt.headers)
                return self._mumt._dataframe[item] # type: ignore
            def __setitem__(self, item: str, value: Any):
                assert item not in (h for umt in self._mumt._umts for h in umt.headers)
                self._mumt.__setitem__(item, value, called_correctly=True)
        return MUMT_S(self)
    
    def __getitem__(self,item):
        #logger.debug(f"Getting column {item} in MultiUniformMeasurementTable via MUMT[col] is deprecated, use MUMT.s[col] or MUMT.h[col] instead.")
        if item in (h for umt in self._umts for h in umt.headers):
            assert len(set(umt.meas_length for umt in self._umts))==1
            import numpy as np
            return np.vstack(self._dataframe[item])
        else:
            return self._dataframe[item]

    def __setitem__(self,item,value, called_correctly: bool = False):
        if not called_correctly:
            pass
            #logger.debug(f"Setting column {item} in MultiUniformMeasurementTable via MUMT[col] is deprecated, use MUMT.s[col] or MUMT.h[col] instead.")
        import pandas as pd
        import numpy as np
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
        import pandas as pd
        import numpy as np
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

    def get_stacked_sweeps(self,only_extr:bool=False) -> pd.DataFrame:
        subs=[]
        prev_meas_id=0
        for umt in self._umts:
            subs.append((ss:=umt.get_stacked_sweeps(only_extr=only_extr)))
            ss['measid']+=prev_meas_id
            prev_meas_id+=len(umt)
        import pandas as pd
        return pd.concat(subs,ignore_index=True)
