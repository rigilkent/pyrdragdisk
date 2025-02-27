import astropy.constants as const
import astropy.units as u

class Star:
    """Configuration parameters for the star.
    Can be initialized directly with parameters or wrap an optprops.Star instance.
    
    :param dist_pc: distance to the observer in parsec
    :param lum_suns: stellar luminosity in solar units (required if optprops_star not provided)
    :param mass_suns: stellar mass in solar masses (required if optprops_star not provided)
    :param optprops_star: optional optprops.Star instance to wrap
    """
    def __init__(self, dist_pc, optprops_star=None, lum_suns=1, mass_suns=1):
        self.dist_pc = dist_pc
        self.dist_cm = (dist_pc * u.pc).to(u.cm).value
        self.dist_au = (dist_pc * u.pc).to(u.au).value
        
        self._optprops_star = optprops_star
        if optprops_star is None:
            self.lum_suns = lum_suns
            self.mass_suns = mass_suns
            
        self.lum_cgs = (self.lum_suns * const.L_sun).to(u.erg/u.s).value
        self.mass_g = (self.mass_suns * const.M_sun).to(u.g).value
        self.gm_cgs = (const.G * self.mass_g * u.g).to(u.cm**3/u.s**2).value

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped optprops.Star object"""
        return getattr(self._optprops_star, name)
