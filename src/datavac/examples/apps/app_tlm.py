#from datavac.examples.filter_plotter_layout import make_allaround_layout
from datavac.examples.apps.app_with_prefilter import PanelAppWithLotPrefilter
from datavac.gui.panel_util.filter_plotter import ScalarFilterPlotter, WafermapFilterPlotter
import param as hvparam

from datavac.gui.plot_templates.tlm_iv import StandardTLMIVPlotter


class AppTLM(PanelAppWithLotPrefilter):
    raw_filter_settings={}
    summ_filter_settings={}
    raw_meas_group=None
    summ_meas_group=None
    normalization_details={}

    vds_options=hvparam.List()
    lsep_name=hvparam.String()

    shownames=hvparam.Dict()
    scalar_plot_pairs=hvparam.Parameter()
    scalar_fig_kwargs=hvparam.Parameter()
    scalar_stars=hvparam.Parameter()
    scalar_categoricals=hvparam.List(default=[])

    wafer_plot_vars=hvparam.Parameter()
    wafer_map=hvparam.Parameter()

    def __init__(self,*args,**kwargs):
        plotters={
            'Raw TLM': StandardTLMIVPlotter(
                #layout_function=make_allaround_layout,
                filter_settings=dict(**self.raw_filter_settings,**self.summ_filter_settings),
                meas_groups=[self.raw_meas_group,self.summ_meas_group],
                normalization_details=self.normalization_details,
                lsep_name=self.lsep_name
            ),
            'Benchmarks': ScalarFilterPlotter(
                #layout_function=make_allaround_layout,
                # TODO: remove goodfit from here.
                # only present right now because summ_filter_settings are also given to Raw
                #filter_settings=dict(**self.summ_filter_settings,**{'goodfit':[True]}),
                filter_settings=self.summ_filter_settings,
                meas_groups=[self.summ_meas_group],
                normalization_details=self.normalization_details,
                stars=self.scalar_stars, shownames=self.shownames,
                plot_pairs=self.scalar_plot_pairs, fig_kwargs=self.scalar_fig_kwargs,
                categoricals=self.scalar_categoricals,
                fig_arrangement='column',
            ),
            'Waferplots': WafermapFilterPlotter(
                #layout_function=make_allaround_layout,
                filter_settings=self.summ_filter_settings,
                meas_groups=[self.summ_meas_group],
                normalization_details=self.normalization_details,
                shownames=self.shownames, plot_vars=self.wafer_plot_vars,
                wmap=self.wafer_map
            ),
        }
        super().__init__(*args,**kwargs,plotters=plotters)
        super().link_shared_widgets()
