import numpy as np
from param import Parameterized
from yaml import YAMLObject, SafeLoader


class MeasurementType(YAMLObject):
    yaml_loader = SafeLoader
    def process_from_raw(self,raw_data_dict):
        return raw_data_dict
    def analyze(self, measurements):
        pass
    def get_preferred_dtype(self,header):
        return np.float32
