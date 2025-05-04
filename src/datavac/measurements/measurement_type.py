import numpy as np
from yaml import YAMLObject, SafeLoader
from typing import TYPE_CHECKING
if TYPE_CHECKING: from datavac.io.measurement_table import UniformMeasurementTable


class MeasurementType(YAMLObject):
    yaml_loader = SafeLoader
    def process_from_raw(self,raw_data_dict: dict[str,np.ndarray]):
        return raw_data_dict
    def analyze(self, measurements: 'UniformMeasurementTable'):
        pass
    def get_preferred_dtype(self, header: str):
        return np.float32
