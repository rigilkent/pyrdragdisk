import numpy as np
import astropy.units as u
import astropy.constants as const
from numpy import pi

c_cgs = const.c.to(u.cm/u.s).value

class Particles:
    """Configuration parameters for particles in a debris disk.

    This class wraps an optprops.Particles instance and adds additional functionality
    specific to debris disk modeling, particularly related to collisional processes.

    Args:
        optprops_prtl: Instance of optprops.Particles containing optical properties
        qd_norm: Strength law scaling parameter (erg/g)
        qd_slope: Strength law slope parameter (must be < 1)
        d_max: Maximum particle size to consider (cm). Defaults to 1.
        k0: Base collision lifetime adjustment factor. Defaults to 4.2.
        gamma: Power law index for collision lifetime adjustment. Defaults to 0.7.
        Qpr: Radiation pressure efficiency. Defaults to 1.
        alpha_r: Slope of redistribution function. Defaults to 3.5.
        diam_min_rel: Minimum relative diameter to consider. Defaults to 1.

    Raises:
        ValueError: If qd_slope >= 1
    """
    def __init__(self, optprops_prtl, qd_norm_cgs=1e7, qd_slope=0, diam_max_cm=float('inf'), 
                 k0=4.2, gamma=0.7, Qpr=1, alpha_r=3.5, diam_min_rel=1):
        if qd_slope >= 1:
            raise ValueError("Parameter 'qd_slope' must be < 1")

        # Store parameters
        self.Qa = qd_norm_cgs
        self.a = np.negative(qd_slope)  # Note: internal parameter 'a' is negative of qd_slope (as in RW20)
        self.k0 = k0
        self.gamma = gamma
        self.Qpr = Qpr
        self.alpha_r = alpha_r
        self.diam_min_rel = diam_min_rel

        # Initialize arrays
        self.k_factor: np.ndarray = None    # Center radii (au)
        # self.diams_blow: tuple = None
        # self.betas: np.ndarray = None

        # Store wrapped instance of optprops.Particles
        self._optprops_prtl = optprops_prtl

        # Filter and convert diameters from optprops_prtl (µm -> cm)
        optprops_prtl.diams *= u.um
        optprops_prtl.diams_blow *= u.um
        mask_diams = optprops_prtl.diams <= diam_max_cm * u.cm
        self.diams = optprops_prtl.diams[mask_diams].to(u.cm).value
        self.diam_max = self.diams[-1]
        self.n_diams = len(self.diams)
        self.diams_blow = optprops_prtl.diams_blow.to(u.cm).value
        self.diam_min = self.diams_blow[1] * diam_min_rel

        # Store relevant optical properties
        self.Qabs = optprops_prtl.Qabs[mask_diams]
        self.Qsca = optprops_prtl.Qsca[mask_diams]
        self.betas = optprops_prtl.betas[mask_diams]

        # Calculate derived properties
        self.calculate_critical_impact_energy()

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped optprops.Particles object"""
        return getattr(self._optprops_prtl, name)

    def calculate_betas_and_blowout_size_simple(self, star, Q_pr=1):
        """Calculates beta values and blowout size(s) for dust grains using a simplified approach,
        that is, assuming a constant radiation pressure efficiency factor Q_pr.

        Args:
            star: A stellar object containing properties like luminosity (lum_cgs) and 
                gravitational parameter (gm_cgs).
            Q_pr (float, optional): Radiation pressure efficiency factor. Defaults to 1.

        Sets:
            - self.betas: Array of beta values for each grain size
            - self.diams_blow: Tuple of (minimum, maximum) blowout grain diameters (cm)
            - self.diam_min: Minimum grain diameter to consider (cm)
        """
        beta_diam_product = 3 * star.lum_cgs * Q_pr / (8 * pi * star.gm_cgs * c_cgs * self.matrl.density)
        self.betas = beta_diam_product / self.diams
        self.diams_blow = (0, beta_diam_product / 0.5)
        self.diam_min = self.diams_blow[1] * self.diam_min_rel

    def calculate_k_factor_tcoll(self):
        """Calculates the k-factor used to adjust collisional lifetimes for different particle sizes.
        The k-factor is computed using the power law relation k = k0 * (d/d_bl)^(-gamma), 
        where d is the particle diameter and d_bl is the blowout diameter.

        Attributes modified:
            k_factor (numpy.ndarray): Array of k-factors for each particle size
        """
        self.k_factor = self.k0 * pow(self.diams / self.diams_blow[1], -self.gamma)

    def interpolate_temperatures(self, rbin):
        """Interpolates temperatures for particles based on radial distances.

        Args:
            rbin: RadialBin object
                Contains the radial bins where temperatures should be interpolated to.
                Must have attributes 'num' (number of bins) and 'mids' (bin midpoints).

        Sets:
            self.temps with interpolated temperature array of shape (rbin.num, self.n_diams).

        Note:
            Original temperature data comes from self._optprops_prtl which must have
              'dists' (distances in au) and 'temps' attributes
        """
        dists_optprops = self._optprops_prtl.dists  # au
        temps_optprops = self._optprops_prtl.temps  # n_dists x n_diams
        temps_new = np.zeros((rbin.num, self.n_diams))
        for iD in range(self.n_diams):  # interpolate in log space
            temps_new[:, iD] = np.interp(np.log10(rbin.mids), np.log10(dists_optprops), np.log10(temps_optprops[:, iD]))
        temps_new = 10**temps_new       # return to lin space
        self.temps = temps_new
 
    def calculate_critical_impact_energy(self):
        """Calculate the critical specific impactor energy at which catastrophic disruption of impactee occurs.
        Corresonding to Rigley & Wyatt (2020), text below Eq. (4).

        Args:
            Qa (float): Normalization parameter of "Q_D star" (erg/g)
            a (float): Slope parameter of "Q_D star" (-)
            diams (float): Particle diameters (cm)

        Returns:
            float: Critical impact energy for dispersal (erg/g)
        """
        self.Q_D = self.Qa * pow(self.diams, -self.a)