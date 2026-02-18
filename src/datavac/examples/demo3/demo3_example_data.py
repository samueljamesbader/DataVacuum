import shutil

from datavac.util.dvlogging import logger


import os
import time

def corrupt_file(lot,sample,meas_name,EXAMPLE_DATA_DIR,subdir=""):
    file_path=EXAMPLE_DATA_DIR/lot/subdir/f"{lot}_{sample}_{meas_name}.csv"

    # Get the original access and modification times
    stat=os.stat(file_path)
    original_atime = stat.st_atime_ns
    original_mtime = stat.st_mtime_ns

    # Edit the file
    with open(file_path, 'w') as file:
        file.write("CORRUPTED DATA")

    # Restore the original timestamps
    os.utime(file_path, ns=(original_atime, original_mtime))
    logger.debug(f"Corrupted file {file_path} but restored original timestamps (ATime: {original_atime}, MTime: {original_mtime})")

def delete_file(lot,sample,meas_name,EXAMPLE_DATA_DIR,subdir=""):
    file_path=EXAMPLE_DATA_DIR/lot/subdir/f"{lot}_{sample}_{meas_name}.csv"
    file_path.unlink()
    logger.debug(f"Deleted file {file_path}")

def write_file(structures, lot, sample, meas_name, EXAMPLE_DATA_DIR, subdir=""):
    from datavac.examples.example_data import get_capacitor, write_example_data_file
    data={structure: get_capacitor('Mask1',structure).generate_potential_cv(VArange=(-1,1),freqs=['1k','10k'], do_plot=False)
          for structure in structures}
    write_example_data_file(lot, sample, meas_name, data, EXAMPLE_DATA_DIR, subdir=subdir)

def ignore_folder(lot, subdir, EXAMPLE_DATA_DIR):
    folder_path=EXAMPLE_DATA_DIR/lot/subdir
    folder_path.rename(folder_path.parent/f"IGNORE_{folder_path.name}")

def make_example_data_demo3_take1():
    from datavac.examples.example_data import get_capacitor, write_example_data_file
    from datavac.examples.demo3 import EXAMPLE_DATA_DIR

    # Clear the directory if it exists and remake it
    if EXAMPLE_DATA_DIR.exists(): shutil.rmtree(EXAMPLE_DATA_DIR)
    EXAMPLE_DATA_DIR.mkdir(parents=True,exist_ok=True)

    write_file(['cap1'],'lot1','sample1', 'A_Cap_CV', EXAMPLE_DATA_DIR)
    write_file(['op1'], 'lot1','sample1','A_Open_CV', EXAMPLE_DATA_DIR)

def change_example_data_demo3_take2():
    from datavac.examples.example_data import get_capacitor, write_example_data_file
    from datavac.examples.demo3 import EXAMPLE_DATA_DIR

    corrupt_file('lot1','sample1','A_Cap_CV',EXAMPLE_DATA_DIR)

    write_file(['cap2'],'lot1','sample1', 'B_Cap_CV', EXAMPLE_DATA_DIR)
    write_file(['op2'], 'lot1','sample1','B_Open_CV', EXAMPLE_DATA_DIR)

    #data={structure: get_capacitor('Mask1',structure).generate_potential_cv(VArange=(-1,1),freqs=['1k','10k'], do_plot=False)
    #      for structure in ['cap3']}
    #write_example_data_file('lot1','sample1','C_Cap_CV',data, EXAMPLE_DATA_DIR)
    #data={structure: get_capacitor('Mask1',structure).generate_potential_cv(VArange=(-1,1),freqs=['1k','10k'], do_plot=False)
    #      for structure in ['op1','op2']}
    #write_example_data_file('lot1','sample1','C_Open_CV',data, EXAMPLE_DATA_DIR)
    #
    #data={structure: get_capacitor('Mask1',structure).generate_potential_cv(VArange=(-1,1),freqs=['1k','10k'], do_plot=False)
    #      for structure in ['cap3']}
    #write_example_data_file('lot1','sample2','C_Cap_CV',data, EXAMPLE_DATA_DIR)
    #data={structure: get_capacitor('Mask1',structure).generate_potential_cv(VArange=(-1,1),freqs=['1k','10k'], do_plot=False)
    #      for structure in ['op1','op2']}
    #write_example_data_file('lot1','sample2','C_Open_CV',data, EXAMPLE_DATA_DIR)

if __name__ == '__main__':
    import os
    os.environ["DATAVACUUM_CONTEXT"]="builtin:demo3"
    make_example_data_demo3_take1()