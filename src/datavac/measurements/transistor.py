
from .measurement_type import MeasurementType

class IdVg(MeasurementType):
    """Assumes headers of the form 'VG', 'fID@VD=...', 'fIG@VD=...', etc, where ... is {VD:.6g}"""
