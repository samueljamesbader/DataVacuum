from panel import Column, Row
from panel.layout import HSpacer
from panel.widgets import TextAreaInput, CrossSelector
import param

class VerticalCrossSelector(CrossSelector):
    _composite_type=Column
    height = param.Integer(default=500, allow_None=True, doc="""
        The number of options shown at once (note this is the
        only way to control the height of this widget)""")
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # Change the buttons to be a row with vertical arrows
        self._buttons[True].name="▼"
        self._buttons[False].name="▲"
        buttons = Row(self._buttons[False], self._buttons[True], margin=(0, 5))

        # Update the placeholder instructions
        layout = dict(sizing_mode='stretch_both', background=self.background, margin=0)
        self._placeholder = TextAreaInput(
            placeholder=("To select an item highlight it above "
                         "and use the arrow button to move down."),
            disabled=True, **layout
        )

        # Remove the "right" (now bottom) filter field
        right = self._lists[True] if self.value else self._placeholder
        self._selected = Column(right, **layout)

        # Make it all
        self._composite[:] = [
            self._unselected, Row(HSpacer(), buttons, HSpacer()), self._selected
        ]