from typing import cast

def test_demo2():

    from datavac.database.db_util import read_sql
    from datavac.config.project_config import PCONF
    from datavac.database.db_create import ensure_clear_database, create_all
    from datavac.database.db_semidev import upload_mask_info
    from datavac.database.db_upload_other import upload_sample_descriptor, upload_subsample_reference
    from datavac.database.db_upload_meas import read_and_enter_data
    from datavac.database.db_get import get_data, get_factors


    from datavac.examples.demo2.demo2_dvconfig import SemiDeviceDataDefinitionFakeLayout
    from datavac.examples.demo2.demo2_dvconfig import get_split_table

    #ddef=cast(SemiDeviceDataDefinitionFakeLayout,PCONF().data_definition)

    ensure_clear_database()
    create_all()
    #upload_mask_info(get_masks())

    upload_sample_descriptor('SplitTable MainFlow', get_split_table())
    #upload_subsample_reference('LayoutParams -- IdVg',ddef.get_layout_params_table('IdVg').reset_index(drop=False))
    read_and_enter_data()
        
    #print(read_sql("""select * from vac."Loads_" """))
    #print(read_sql("""select * from vac."Meas -- IdVg" """))
    #print(read_sql("""select * from vac."ReLoad_" """))
    #print(read_sql("""select * from vac."Extr -- IdVg" """))
    #print(read_sql("""select * from jmp."IdVg" """))
    assert get_factors('IdVg',['Structure'],pre_filters={'Structure': ['nMOS1']})=={'Structure': {'nMOS1'}}
    assert get_factors('IdVg',['W [um]'],pre_filters={'Structure': ['nMOS1']})=={'W [um]': {1.0}}
    assert get_factors('IdVg',['W [um]'])=={'W [um]': {1.0, 2.0}}
    assert get_factors('IdVg',['Structure','W [um]'])=={'Structure':{'nMOS1','nMOS2'},'W [um]': {1.0, 2.0}}
    


    #for sample, data_by_mg in sample_to_mg_to_data.items():
    #    upload_measurement(trove, sample_to_sampleloadinfo[sample], data_by_mg)
    #print(read_sql("""select * from vac."ReLoad_" """))

if __name__ == '__main__':
    import os
    os.environ["DATAVACUUM_CONTEXT"]="builtin:demo2"
    test_demo2()