from typing import Optional
from datavac.config.data_definition import DDEF
from datavac.io.measurement_table import MultiUniformMeasurementTable
from datavac.util.dvlogging import logger
from datavac.util.tables import check_dtypes

def perform_extraction(matname_to_mg_to_data: dict[str,dict[str,MultiUniformMeasurementTable]],
                       only_meas_groups:Optional[list[str]]=None) -> None:
    for matname, mg_to_data in matname_to_mg_to_data.items():
        to_be_extracted=only_meas_groups or list(mg_to_data.keys())
        to_be_extracted = [mg.name for mg in DDEF().get_meas_groups_topo_sorted() if mg.name in to_be_extracted]

        for mg_name in to_be_extracted:
            from datavac.config.project_config import PCONF
            mg=PCONF().data_definition.measurement_groups[mg_name]
            deps=dict(**mg.required_dependencies,**mg.optional_dependencies)

            data=mg_to_data[mg_name]
            logger.debug(f"{mg_name} extraction ({matname})")
            dep_kws={deps[d]:mg_to_data[d] for d in deps if d in mg_to_data}
            mg.extract_by_mumt(data, **dep_kws)

            # TODO: this check is assembling the full MUMT in memory, which is unnecessary just to get columns...
            ######cols=data._dataframe.columns
            ######for k in mg.extr_column_names: assert k in cols
            ######check_dtypes(data.scalar_table)
