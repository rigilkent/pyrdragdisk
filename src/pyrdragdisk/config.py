import warnings
from .warnings import AssumedUnitWarning

def set_assumed_unit_warnings(enable: bool = True, *, once: bool = True) -> None:
    """
    Globally enable/disable warnings when unitless inputs get assumed units.
    """
    # Clear existing filters for this warning category
    warnings.filterwarnings("ignore", category=AssumedUnitWarning)

    if enable:
        action = "once" if once else "default"
        warnings.filterwarnings(action, category=AssumedUnitWarning)
