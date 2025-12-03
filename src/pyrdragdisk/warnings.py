"""
pyrdragdisk.warnings
-------------------

Central place for all package-specific warning classes.
"""

class PyrDragDiskWarning(UserWarning):
    """Base class for all pyrdragdisk warnings."""
    pass


class AssumedUnitWarning(PyrDragDiskWarning):
    """Warning raised when a unitless value is interpreted with an assumed unit."""
    pass


import warnings as _warnings

def warn_assumed_unit(param_name: str, assumed_unit, *, extra: str = "") -> None:
    """
    Emit a standardized warning about an assumed unit for a parameter.

    Parameters
    ----------
    param_name : str
        Name of the parameter (e.g. "radius").
    assumed_unit : astropy.units.UnitBase or str
        The unit that was assumed.
    extra : str, optional
        Extra context to append to the message.
    """
    msg = (
        f"Unitless value passed for parameter '{param_name}'; "
        f"assuming units of {assumed_unit}."
    )
    if extra:
        msg += " " + extra

    _warnings.warn(msg, AssumedUnitWarning, stacklevel=2)