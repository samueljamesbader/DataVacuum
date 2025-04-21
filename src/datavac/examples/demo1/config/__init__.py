def complete_matload_info(matload_info:dict):
    matload_info=matload_info.copy()
    matload_info['Mask']=matload_info.get('Mask','Mask1')
    return matload_info