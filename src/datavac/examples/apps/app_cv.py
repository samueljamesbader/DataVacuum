import param as hvparam
from datavac.examples.apps.app_with_prefilter import PanelAppWithLotPrefilter
from datavac.examples.apps.filter_plotter_layout import make_allaround_layout
from datavac.gui.panel_util.filter_plotter import ScalarFilterPlotter, WafermapFilterPlotter
from datavac.gui.plot_templates.diode_dciv import StandardDCIVPlotter
from datavac.gui.plot_templates.xtor_cv import StandardCVPlotter
class AppCV(PanelAppWithLotPrefilter):
    filter_settings={}
    meas_groups={}
    normalization_details={}
    freqs_options=hvparam.List()
    shift_by_options=hvparam.Selector()

    shownames=hvparam.Dict()

    scalar_plot_pairs=hvparam.Parameter()
    scalar_fig_kwargs=hvparam.Parameter()
    scalar_stars=hvparam.Parameter({})
    scalar_categoricals=hvparam.List(default=[])
    def __init__(self,*args,**kwargs):
        plotters={
            'Raw CV': StandardCVPlotter(
                layout_function=make_allaround_layout,
                filter_settings=self.filter_settings,
                meas_groups=self.meas_groups,
                normalization_details=self.normalization_details,
                shift_by_options=self.shift_by_options,freqs_options=self.freqs_options,
            ),
            'Benchmarks': ScalarFilterPlotter(
                layout_function=make_allaround_layout,
                filter_settings=self.filter_settings,
                meas_groups=self.meas_groups,
                normalization_details=self.normalization_details,
                plot_pairs=self.scalar_plot_pairs,stars=self.scalar_stars,
                fig_kwargs=self.scalar_fig_kwargs, shownames=self.shownames,
                categoricals=self.scalar_categoricals,
            ),
        }
        super().__init__(*args,**kwargs,plotters=plotters)
        super().link_shared_widgets()

class AppCVIV(PanelAppWithLotPrefilter):
    filter_settings={}
    cv_meas_group=None
    iv_meas_group=None
    normalization_details={}
    freqs_options=hvparam.List()
    shift_by_options=hvparam.Selector()

    shownames=hvparam.Dict()

    scalar_plot_pairs=hvparam.Parameter()
    scalar_fig_kwargs=hvparam.Parameter()
    scalar_stars=hvparam.Parameter({})
    scalar_categoricals=hvparam.List(default=[])

    wafer_plot_vars = None
    wafer_plot_vars_to_meas_group = None

    def __init__(self,*args,**kwargs):
        self.lot_prefetch_measgroup=[self.cv_meas_group,self.iv_meas_group,self.iv_meas_group]
        plotters={
            'Raw CV': StandardCVPlotter(
                layout_function=make_allaround_layout,
                filter_settings=self.filter_settings,
                meas_groups=[self.cv_meas_group],
                normalization_details=self.normalization_details,
                shift_by_options=self.shift_by_options,freqs_options=self.freqs_options,
            ),
            'Raw IV': StandardDCIVPlotter(
                layout_function=make_allaround_layout,
                filter_settings=self.filter_settings,
                meas_groups=[self.iv_meas_group],
                normalization_details=self.normalization_details,
            ),
            #'Benchmarks': ScalarFilterPlotter(
            #    layout_function=make_allaround_layout,
            #    filter_settings=self.filter_settings,
            #    meas_groups=self.meas_groups,
            #    normalization_details=self.normalization_details,
            #    plot_pairs=self.scalar_plot_pairs,stars=self.scalar_stars,
            #    fig_kwargs=self.scalar_fig_kwargs, shownames=self.shownames,
            #    categoricals=self.scalar_categoricals,
            #),
        }
        if self.wafer_plot_vars:
            plotters['Waferplots']= WafermapFilterPlotter(
                layout_function=make_allaround_layout,
                filter_settings=self.filter_settings,
                #meas_groups=self.meas_groups,
                normalization_details=self.normalization_details,
                shownames=self.shownames, plot_vars=self.wafer_plot_vars,
                plot_var_to_meas_group=self.wafer_plot_vars_to_meas_group,
                wmap=self.wafer_map, meas_groups=None
            )
        super().__init__(*args,**kwargs,plotters=plotters)
        super().link_shared_widgets(except_params=['bin'])
