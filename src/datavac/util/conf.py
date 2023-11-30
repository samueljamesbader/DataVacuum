import os
from pathlib import Path

import yaml

from datavac.util.util import import_modfunc


class Config():

    def __init__(self):
        with open(Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"project.yaml",'r') as f:
            self._yaml=yaml.safe_load(f)

    def __getattr__(self, item):
        return self._yaml[item]

    def __getitem__(self, item):
        return self._yaml[item]

    def get_meas_type(self, meas_group):
        res=self.measurement_groups[meas_group]['meas_type']
        if type(res) is str:
            return import_modfunc(res)()
        else:
            return import_modfunc(res[0])(**res[1])

    def get_dependent_analyses(self, meas_groups):
        return list(set(an for an,an_info in self.higher_analyses.items()\
            if any(mg in meas_groups for mg in
                   list(an_info.get('required_dependencies',{}).keys())+ \
                   list(an_info.get('attempt_dependencies',{}).keys()))))

    def get_dependency_meas_groups_for_analyses(self, analyses, required_only=False):
        if required_only:
            return list(set([mg for an in analyses
                 for mg in self.higher_analyses[an]['required_dependencies']]))
        else:
            return list(set([mg for an in analyses
                 for mg in list(self.higher_analyses[an]['required_dependencies'].keys())+\
                             list(self.higher_analyses[an].get('attempt_dependencies',{}).keys())]))

    def get_dependency_meas_groups_for_meas_groups(self, meas_groups, required_only=False):
        if required_only:
            return list(set([mg for mg_ in meas_groups for mg in
                             list(self.measurement_groups[mg_].get('required_dependences',{}).keys())]))
        else:
            return list(set([mg for mg_ in meas_groups for mg in
                             list(self.measurement_groups[mg_].get('required_dependences',{}).keys())+ \
                             list(self.measurement_groups[mg_].get('attempt_dependences',{}).keys())]))

CONFIG=Config()