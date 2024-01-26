from datavac.examples.apps.filter_plotter_layout import make_allaround_layout
from datavac.examples.apps.app_with_prefilter import PanelAppWithLotPrefilter
from datavac.gui.plot_templates.xtor_idvd import StandardIdVdPlotter
import param as hvparam
class AppIdVd(PanelAppWithLotPrefilter):
    filter_settings={}
    meas_groups={}
    normalization_details={}

    pol=hvparam.Selector(objects=["p","n"])
    vgs_options=hvparam.List()

    shownames=hvparam.Dict()
    scalar_plot_pairs=hvparam.Parameter()
    scalar_fig_kwargs=hvparam.Parameter()
    scalar_stars=hvparam.Parameter()
    scalar_categoricals=hvparam.List(default=[])

    wafer_plot_vars=hvparam.Parameter()
    wafer_map=hvparam.Parameter()

    def __init__(self,*args,**kwargs):
        plotters={
            'Raw IdVd': StandardIdVdPlotter(
                layout_function=make_allaround_layout,
                filter_settings=self.filter_settings,
                meas_groups=self.meas_groups,
                normalization_details=self.normalization_details,
                pol=self.pol,vgs_options=self.vgs_options,
            ),
            #'Benchmarks': ScalarFilterPlotter(
            #    layout_function=make_allaround_layout,
            #    filter_settings=self.filter_settings,
            #    meas_groups=self.meas_groups,
            #    normalization_details=self.normalization_details,
            #    stars=self.scalar_stars, shownames=self.shownames,
            #    plot_pairs=self.scalar_plot_pairs, fig_kwargs=self.scalar_fig_kwargs,
            #    categoricals=self.scalar_categoricals,
            #),
            #'Waferplots': WafermapFilterPlotter(
            #    layout_function=make_allaround_layout,
            #    filter_settings=self.filter_settings,
            #    meas_groups=self.meas_groups,
            #    normalization_details=self.normalization_details,
            #    shownames=self.shownames, plot_vars=self.wafer_plot_vars,
            #    wmap=self.wafer_map
            #),
        }
        super().__init__(*args,**kwargs,plotters=plotters)
        super().link_shared_widgets()
