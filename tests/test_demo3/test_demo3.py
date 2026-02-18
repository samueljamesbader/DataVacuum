
def test_demo3():
    import shutil
    import numpy as np
    from datavac.database.db_upload_meas import read_and_enter_data
    from datavac.examples.demo3.demo3_example_data import make_example_data_demo3_take1, change_example_data_demo3_take2
    from datavac.database.db_get import get_data
    from datavac.examples.demo3.demo3_example_data import write_file, corrupt_file, delete_file, ignore_folder

    from datavac.examples.demo3 import EXAMPLE_DATA_DIR
    # Clear the directory if it exists and remake it
    if EXAMPLE_DATA_DIR.exists(): shutil.rmtree(EXAMPLE_DATA_DIR)
    EXAMPLE_DATA_DIR.mkdir(parents=True,exist_ok=True)
    
    # Make A with cap1
    write_file(['cap1'],'lot1','sample1','f1_A_Cap_CV' , EXAMPLE_DATA_DIR)
    write_file(['op1'], 'lot1','sample1','f1_A_Open_CV', EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('A_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[1e-14], rtol=0.001,atol=0)
    x=get_data('B_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==0
    x=get_data('A_Open_CV').sort_values('FilePath')['Cmean [F]'];print(x);assert len(x)==1 and np.allclose(x,[2e-15], rtol=0.001,atol=0)

    # Corrupt A and add B with cap2
    corrupt_file('lot1','sample1','f1_A_Cap_CV',EXAMPLE_DATA_DIR)
    corrupt_file('lot1','sample1','f1_A_Open_CV',EXAMPLE_DATA_DIR)
    write_file(['cap2'],'lot1','sample1','f2_B_Cap_CV' , EXAMPLE_DATA_DIR)
    write_file(['op2'], 'lot1','sample1','f2_B_Open_CV', EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('A_Cap_CV').sort_values('FilePath')['Cmean [F]'];print(x);assert len(x)==1 and np.allclose(x,[1e-14], rtol=0.001,atol=0)
    x=get_data('B_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[2e-14], rtol=0.001,atol=0)
    x=get_data('A_Open_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[2e-15], rtol=0.001,atol=0)

    # Corrupt B and add more A with cap2
    corrupt_file('lot1','sample1','f2_B_Cap_CV',EXAMPLE_DATA_DIR)
    corrupt_file('lot1','sample1','f2_B_Open_CV',EXAMPLE_DATA_DIR)
    write_file(['cap2'],'lot1','sample1','f3_A_Cap_CV' , EXAMPLE_DATA_DIR)
    write_file(['op2'], 'lot1','sample1','f3_A_Open_CV', EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('A_Cap_CV').sort_values('FilePath')['Cmean [F]'];print(x);assert len(x)==2 and np.allclose(x,[1e-14,2e-14], rtol=0.001,atol=0)
    x=get_data('B_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[2e-14], rtol=0.001,atol=0)
    x=get_data('A_Open_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==2 and np.allclose(x,[2e-15,3e-15], rtol=0.001,atol=0)

    # Overwrite A's cap2 with cap3
    write_file(['cap3'],'lot1','sample1','f3_A_Cap_CV' , EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('A_Cap_CV').sort_values('FilePath')['Cmean [F]'];print(x);assert len(x)==2 and np.allclose(x,[1e-14,3e-14], rtol=0.001,atol=0)
    x=get_data('B_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[2e-14], rtol=0.001,atol=0)
    x=get_data('A_Open_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==2 and np.allclose(x,[2e-15,3e-15], rtol=0.001,atol=0)

    # Delete A's cap3 (ie some of A cap data but not all of it)
    delete_file('lot1','sample1','f3_A_Cap_CV', EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('A_Cap_CV').sort_values('FilePath')['Cmean [F]'];print(x);assert len(x)==1 and np.allclose(x,[1e-14], rtol=0.001,atol=0)
    x=get_data('B_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[2e-14], rtol=0.001,atol=0)
    x=get_data('A_Open_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==2 and np.allclose(x,[2e-15,3e-15], rtol=0.001,atol=0)

    # Add a new sample for A
    write_file(['cap1'],'lot1','sample2','f4_A_Cap_CV' , EXAMPLE_DATA_DIR)
    write_file(['op1'], 'lot1','sample2','f4_A_Open_CV', EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('A_Cap_CV').sort_values(['FilePath','LotSample'])[['Cmean [F]','LotSample']];print(x);assert len(x)==2 and np.allclose(x['Cmean [F]'],[1e-14,1e-14], rtol=0.001,atol=0) and set(x['LotSample'])==set(['lot1_sample1','lot1_sample2'])
    x=get_data('B_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[2e-14], rtol=0.001,atol=0)
    x=get_data('A_Open_CV').sort_values('FilePath')['Cmean [F]'];print(x);assert len(x)==3 and np.allclose(x,[2e-15,3e-15,2e-15], rtol=0.001,atol=0)

    # Delete the open for the new sample from A (ie all the open data for the new sample)
    delete_file('lot1','sample2','f4_A_Open_CV', EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('A_Cap_CV').sort_values(['FilePath','LotSample'])[['Cmean [F]','LotSample']];print(x);assert len(x)==2 and np.allclose(x['Cmean [F]'],[1e-14,1.2e-14], rtol=0.001,atol=0) and set(x['LotSample'])==set(['lot1_sample1','lot1_sample2'])
    x=get_data('B_Cap_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==1 and np.allclose(x,[2e-14], rtol=0.001,atol=0)
    x=get_data('A_Open_CV').sort_values('FilePath')['Cmean [F]'];assert len(x)==2 and np.allclose(x,[2e-15,3e-15], rtol=0.001,atol=0)

    # Add new sample for C in a subdir "other"
    write_file(['cap3'],'lot1','sample3','f5_C_Cap_CV' , EXAMPLE_DATA_DIR, subdir="other")
    write_file(['op2'], 'lot1','sample3','f5_C_Open_CV', EXAMPLE_DATA_DIR, subdir="other")
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('C_Cap_CV').sort_values(['FilePath','LotSample'])[['Cmean [F]','LotSample']];print(x);assert len(x)==1 and np.allclose(x['Cmean [F]'],[3e-14], rtol=0.001,atol=0) and set(x['LotSample'])==set(['lot1_sample3'])
    x=get_data('C_Open_CV').sort_values(['FilePath','LotSample'])[['Cmean [F]','LotSample']];print(x);assert len(x)==1 and np.allclose(x['Cmean [F]'],[3e-15], rtol=0.001,atol=0) and set(x['LotSample'])==set(['lot1_sample3'])

    # Ignore "other"
    ignore_folder('lot1','other', EXAMPLE_DATA_DIR)
    read_and_enter_data(kwargs_by_trove={'':{'dont_prompt_readall':True}}, suspend_exceptions=False, incremental=True)
    x=get_data('C_Cap_CV').sort_values(['FilePath','LotSample'])[['Cmean [F]','LotSample']];print(x);assert len(x)==0
    x=get_data('C_Open_CV').sort_values(['FilePath','LotSample'])[['Cmean [F]','LotSample']];print(x);assert len(x)==0

if __name__ == '__main__':
    import os
    os.environ["DATAVACUUM_CONTEXT"]="builtin:demo3"
    # Note: for imports to work run as "python -m tests.test_demo3.test_demo3" from the DataVacuum directory
    from tests.test_demo3.conftest import _prep_db
    _prep_db()
    test_demo3()