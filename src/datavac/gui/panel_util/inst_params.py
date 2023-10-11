from panel.widgets.base import CompositeWidget
from param.parameterized import ParameterizedMetaclass

class CompositeWidgetWithInstanceParameters(CompositeWidget):
    # The param package associates the set of parameters with *classes* not *instances*.
    # Even if it's a per-instance parameter (ie it can have a different value for each instance), the fact that the
    # parameter exists at all is a class-level attribute... So when I have a case where I want the set of parameters
    # for a FilterPlotter instance to be supplied at run-time, I need each FilterPlotter instance to really be its
    # own class, otherwise an instance on one will appear on all.  [Note: I don't actually think this would have any
    # side-effects so far, but it's still weird.]  But it would be a pain to make all those subclasses separately,
    # so instead I'll override `__new__`, such that each instance `I` of a FilterPlotter subclass `S` gets its own
    # instance-specific class `type(I)==S2` where `S2` is a dynamically generated subclass of `S`.  Now parameters added
    # to `I` will appear on `S2` but not on `S`, so they don't affect other instances.
    def __new__(cls, *args, **kwargs):
        SpecificWidget=ParameterizedMetaclass(f"Specific{cls.__name__}",(cls,),{})
        return super(CompositeWidgetWithInstanceParameters,SpecificWidget).__new__(SpecificWidget)
