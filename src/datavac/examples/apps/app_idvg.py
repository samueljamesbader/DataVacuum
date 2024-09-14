from functools import partial

from datavac.examples.apps.filter_plotter_layout import make_allaround_layout
from datavac.examples.apps.app_with_prefilter import PanelAppWithLotPrefilter
from datavac.gui.panel_util.filter_plotter import ScalarFilterPlotter, WafermapFilterPlotter
from datavac.gui.plot_templates.xtor_idvg import StandardIdVgPlotter
import param as hvparam
class AppIdVg(PanelAppWithLotPrefilter):
    filter_settings={}
    meas_groups={}
    normalization_details={}

    show_raw=hvparam.Boolean(default=True)
    pol=hvparam.Selector(objects=["p","n"])
    vds_options=hvparam.List()

    shownames=hvparam.Dict()
    scalar_plot_pairs=hvparam.Parameter()
    scalar_fig_kwargs=hvparam.Parameter()
    scalar_stars=hvparam.Parameter()
    scalar_categoricals=hvparam.List(default=[])
    scalar_fig_arrangement=hvparam.String("row")
    scalar_main_row_height=hvparam.Parameter(None)

    wafer_plot_vars=hvparam.Parameter()
    wafer_map=hvparam.Parameter()


    def __init__(self,*args,**kwargs):
        plotters={}
        if self.show_raw:
            plotters['Raw IdVg']= StandardIdVgPlotter(
                    layout_function=make_allaround_layout,
                    filter_settings=self.filter_settings,
                    meas_groups=self.meas_groups,
                    normalization_details=self.normalization_details,
                    pol=self.pol,vds_options=self.vds_options,
                )
        if self.scalar_plot_pairs:
            plotters['Benchmarks']= ScalarFilterPlotter(
                layout_function=partial(make_allaround_layout,
                    **{k:v for k,v in {'main_row_height':self.scalar_main_row_height}.items() if v is not None}),
                filter_settings=self.filter_settings,
                meas_groups=self.meas_groups,
                normalization_details=self.normalization_details,
                stars=self.scalar_stars, shownames=self.shownames,
                plot_pairs=self.scalar_plot_pairs, fig_kwargs=self.scalar_fig_kwargs,
                categoricals=self.scalar_categoricals, fig_arrangement=self.scalar_fig_arrangement
            )
        if self.wafer_plot_vars:
            plotters['Waferplots']= WafermapFilterPlotter(
                layout_function=make_allaround_layout,
                filter_settings=self.filter_settings,
                meas_groups=self.meas_groups,
                normalization_details=self.normalization_details,
                shownames=self.shownames, plot_vars=self.wafer_plot_vars,
                wmap=self.wafer_map
            )
        super().__init__(*args,**kwargs,plotters=plotters)
        super().link_shared_widgets()
