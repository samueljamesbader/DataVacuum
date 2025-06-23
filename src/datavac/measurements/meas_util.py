from datavac.io.measurement_table import MultiUniformMeasurementTable
from datavac.util.logging import logger
from datavac.util.tables import check_dtypes

def perform_extraction(matname_to_mg_to_data: dict[str,dict[str,MultiUniformMeasurementTable]]) -> None:
    for matname, mg_to_data in matname_to_mg_to_data.items():
        to_be_extracted=list(mg_to_data.keys())
        #ensure_meas_group_sufficiency(to_be_extracted,required_only=True, just_extraction=True)
        logger.critical("Would normally ensure measurement group sufficiency here, but skipping for now")
        while len(to_be_extracted):
            for mg_name in to_be_extracted:
                from datavac.config.project_config import PCONF
                mg=PCONF().data_definition.measurement_groups[mg_name]
                deps=dict(**mg.required_dependencies,**mg.optional_dependencies)
                if any(d in to_be_extracted for d in deps): continue

                data=mg_to_data[mg_name]
                logger.debug(f"{mg_name} extraction ({matname})")
                dep_kws={deps[d]:mg_to_data[d] for d in deps if d in mg_to_data}
                #data.analyze(**dep_kws)
                mg.extract_by_mumt(data, **dep_kws)

                # TODO: this check is assembling the full MUMT in memory, which is unnecessary just to get columns...
                cols=data._dataframe.columns
                for k in mg.extr_column_names: assert k in cols
                check_dtypes(data.scalar_table)

                to_be_extracted.remove(mg_name)