import functools
from typing import Optional
from datavac.database.db_structure import DBSTRUCT, sql_to_pd_types
from datavac.measurements.measurement_group import MeasurementGroup
from sqlalchemy import select, Select, Column, Table

def joined_select_from_dependencies(columns:Optional[list[str]],absolute_needs:list[Table],
                 table_depends:dict[Table,list[Table]], pre_filters:dict[str,list],
                 join_hints:dict[Table,str]={}) -> tuple[Select, dict[str, str]]:
    """ Creates an SQL Alchemy Select joining the tables needed to get the desired factors

    Args:
        columns: list of column names to be selected (or None to select all columns from table_depends)
        absolute_needs: list of tables which must be included in the join
        table_depends: mapping of dependencies which tables have on other tables to be joined
        pre_filters: filters (column name to list of allowed values) to apply to the data 
        join_hints: mapping of table to join clause to use when joining that table
            (generally not needed; if tables have foreign keys, sqlalchemy will infer)

    Returns:
        An SQLAlchemy Select and a dictionary mapping column names to their intended pandas types
    """
    all_cols=[c for tab in table_depends for c in tab.columns]
    def get_col(cname) -> Column:
        try: return next(c for c in all_cols if c.name==cname)
        except StopIteration:
            raise Exception(f"Couldn't find column {cname} among {[c.name for c in all_cols]}")
    def apply_wheres(s):
        for pf,values in pre_filters.items():
            s=s.where(get_col(pf).in_(values))
        return s
    if columns is not None:
        cols:list[Column]=[get_col(f) for f in columns]
    else:
        col_names = set()
        cols: list[Column] = []
        for tab in table_depends:
            for c in tab.columns:
                if c.name not in col_names:
                    cols.append(c); col_names.add(c.name)
    def apply_joins():
        ordered_needed_tables=[]
        need_queue=set(absolute_needs+[f.table for f in cols]+[get_col(pf).table for pf in pre_filters])
        while len(need_queue):
            for n in need_queue.copy():
                further_needs=table_depends[n]
                if n in ordered_needed_tables:
                    need_queue.remove(n)
                elif all(pn in ordered_needed_tables for pn in further_needs):
                    ordered_needed_tables.append(n)
                    need_queue.remove(n)
                else: need_queue|=set(further_needs)
        return functools.reduce((lambda x,y: x.join(y,onclause=join_hints.get(y,None))),ordered_needed_tables)
    sel=apply_wheres(select(*cols).select_from(apply_joins()))
    dtypes={c.name:sql_to_pd_types[c.type.__class__] for c in cols
                                             if c.type.__class__ in sql_to_pd_types}
    return sel, dtypes

def get_table_depends_and_hints_for_meas_group(meas_group: MeasurementGroup):
        trove_name = meas_group.trove_name()

        table_depends={}
        table_depends[coretab:=(meastab:=DBSTRUCT().get_measurement_group_dbtables(meas_group.name)['meas'])]=[]
        table_depends[         (extrtab:=DBSTRUCT().get_measurement_group_dbtables(meas_group.name)['extr'])]=[meastab]
        table_depends[loadtab:=(DBSTRUCT().get_trove_dbtables(trove_name)['loads'])]=[coretab]
        table_depends[sampletab:=(DBSTRUCT().get_sample_dbtable())]=[loadtab]
        for ssr_name in meas_group.subsample_reference_names:
            # Note: just including the sample table here because in case the subsample reference has foreign keys
            # to some sample info, we need to make sure the sample info is merged before the subsample reference
            table_depends[DBSTRUCT().get_subsample_reference_dbtable(ssr_name)]=[coretab,DBSTRUCT().get_sample_dbtable()]

        # TODO: Uncomment when fnct and ot readded 
        #fnc_tables=[self._hat(fnct) for fnct in fnc_tables]
        #other_tables=[self._hat(ot) for ot in other_tables]
        #for fnct in fnc_tables: table_depends[fnct]=[self._mattab]
        #for ot in other_tables: table_depends[ot]=[self._mattab]

        return table_depends, {}
    

#def get_view(meas_group: MeasurementGroup) -> Select:
#
#        trove_name = meas_group.trove_name()
#        # Tables we may need
#        table_depends={}
#        table_depends[coretab:=(meastab:=DBSTRUCT().get_measurement_group_dbtables(meas_group.name)['meas'])]=[]
#        table_depends[         (extrtab:=DBSTRUCT().get_measurement_group_dbtables(meas_group.name)['extr'])]=[meastab]
#        #if which == 'higher_analyses':
#        #    table_depends[coretab:=(anlstab:=self._hat(mgoa))]=[]
#        table_depends[loadtab:=(DBSTRUCT().get_trove_dbtables(trove_name))]=[coretab]
#        table_depends[sampletab:=(DBSTRUCT().get_sample_dbtable())]=[loadtab]
#        for ssr_name in meas_group.subsample_reference_names:
#            # Note: just including the sample table here because in case the subsample reference has foreign keys
#            # to some sample info, we need to make sure the sample info is merged before the subsample reference
#            table_depends[DBSTRUCT().get_subsample_reference_dbtable(ssr_name)]=[coretab,DBSTRUCT().get_sample_dbtable()]
#
#        fnc_tables=[self._hat(fnct) for fnct in fnc_tables]
#        other_tables=[self._hat(ot) for ot in other_tables]
#        for fnct in fnc_tables: table_depends[fnct]=[self._mattab]
#        for ot in other_tables: table_depends[ot]=[self._mattab]
#
#
#        # Join hints
#        join_hints={fnct:(fnct.c[fullname_col]==self._mattab.c[fullname_col]) for fnct in fnc_tables}
#        if which == 'higher_analyses':
#            join_hints[self._loadtab]=(anlstab.c[f"loadid - {list(CONFIG.higher_analyses[mgoa]['required_dependencies'])[0]}"]==self._loadtab.c.loadid)
#
#        return joined_select_from_dependencies(columns=factor_names,absolute_needs=[coretab],
#                            table_depends=table_depends,pre_filters=pre_filters,
#                            join_hints=join_hints)[0]