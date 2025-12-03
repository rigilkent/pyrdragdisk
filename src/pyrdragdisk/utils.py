import astropy.units as u
from .warnings import warn_assumed_unit

# -----------------
# Quantity handling
# -----------------
def as_quantity(value, unit, *, param_name):
    if isinstance(value, u.Quantity):
        return value.to(unit)
    else:
        warn_assumed_unit(param_name, unit)
        return value * unit

def as_value(value, unit, *, param_name):
    """Return the value converted to the given unit, stripped of units."""
    return as_quantity(value, unit, param_name=param_name).value
