from bokeh.models import ColumnDataSource

from datavac.gui.panel_util.filter_plotter import FilterPlotter


class StandardXtorSParVFreqPlotter(FilterPlotter):
    _built_in_view_settings = ['norm_by','color_by','vds']
    vds=hvparam.ListSelector()
    def __init__(self,vds_options,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # Set options and defaults for the view settings
        self.param.color_by.objects=list(self.filter_settings.keys())+['VD','SweepDir']
        if self.color_by is None: self.color_by='LotWafer'

    def get_raw_column_names(self):
        return [['freq']+[f'{sd}I{term}@VD={vd}' for vd in self.param.vds.objects for term in ['G','D'] for sd in self.param.sweep_dirs.objects]]

    def update_sources(self, pre_sources, event=None):

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'curves':ColumnDataSource({})}

            # Empty prototype
            self._sources['curves'].data={'VG':[],'ID':[],'IG':[],'GM':[], 'legend':[], 'color':[]}
            self._sources['ylabels']=None

        # Otherwise, analyze the real data
        else:
            pass
