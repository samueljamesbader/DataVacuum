import panel as pn

from datavac.gui.panel_util.filter_plotter import FilterPlotter


def make_allaround_layout(filter_widgets, view_widgets, figure_pane):
    top_filter_widgets={k:v for k,v in filter_widgets.items()\
                        if k not in ['Structure','LotWafer','DieLB','FileName']}
    side_filter_widgets={k:v for k,v in filter_widgets.items() \
                        if k in ['Structure','LotWafer','DieLB']}
    bottom_filter_widgets={k:v for k,v in filter_widgets.items() \
                         if k in ['FileName']}
    side_filter_widgets['LotWafer'].sizing_mode='stretch_height'
    side_filter_widgets['LotWafer'].width=140
    side_filter_widgets['DieLB'].sizing_mode='stretch_height'
    side_filter_widgets['DieLB'].width=80
    side=pn.Column(
        side_filter_widgets['Structure'],
        pn.Row(side_filter_widgets['LotWafer'],side_filter_widgets['DieLB']),
        width=250, height=300
    )
    top_row=pn.Row(*top_filter_widgets.values(),
                   sizing_mode='stretch_width',height=125)
    main_row=pn.Row(side,figure_pane,#pn.HSpacer(),
                    pn.Column(*view_widgets.values(),sizing_mode='stretch_both'),
                    sizing_mode='stretch_width',height=300)
    bottom_row=pn.Row(*bottom_filter_widgets.values(),
                    sizing_mode='stretch_width',height=300)
    return pn.Column(top_row,pn.HSpacer(height=10),main_row,bottom_row,sizing_mode='stretch_height',width=1200)
