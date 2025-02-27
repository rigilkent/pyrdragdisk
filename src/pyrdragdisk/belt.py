import numpy as np
import astropy.units as u
import astropy.constants as const
from numpy import pi

# Pre-compute CGS values for common conversions
c_cgs = const.c.to(u.cm/u.s).value
year_seconds = u.yr.to(u.s)

class Belt:
    """Configuration parameters for debris disk belt."""
    def __init__(self, r0_au, dr_r, inc_max_deg, m_dust_earths, avg_ecc=0):
        # Basic geometric parameters
        self.r0_au = r0_au
        self.r0_cm = (r0_au * u.au).to(u.cm).value
        self.dr_r = dr_r
        self.r_out_au = self.r0_au * (1 + self.dr_r)
        self.r_out_cm = (self.r_out_au * u.au).to(u.cm).value
        self.inc_max_deg = inc_max_deg
        self.inc_max_rad = np.radians(inc_max_deg)
        self.avg_ecc = avg_ecc

        # Mass parameters
        self.m_dust_earths = m_dust_earths
        self.m_dust_g = (m_dust_earths * const.M_earth).to(u.g).value

        # Derived geometric parameters
        self.vol = self.calculate_volume()

        # All following belt parameters also depend on star and particle properties
        # and are calculated by the calculate_properties() method.
        self.vel_coll: float = None           # Collisional velocity (cm/s)
        self.X_C: np.ndarray = None           # Critical impactor size ratio (-)
        self.t_coll_k: np.ndarray = None      # Collision timescale (yr)
        
        # Parameters calculated by calculate_size_distribution_both_regimes()
        self.n_D: np.ndarray = None     # Size distribution (-)
        self.alpha: float = None        # Size distribution slope (-)
        self.D_pr: float = None         # PR-drag regime turnover size (cm)
        self.K_norm: float = None       # Normalization factor for collisional regime
        self.K_norm_pr: float = None    # Normalization factor for PR-drag regime
        
        # Parameters calculated by calculate_ratio_lifetimes_prdrag_to_coll()
        self.eta_0: np.ndarray = None   # Ratio of PR-drag to collisional timescales (-)
        self.D_pr_eff: float = None     # Effective turnover size (cm)
        self.t_PR: np.ndarray = None    # PR-drag timescale (yr)

    def calculate_volume(self) -> float:
        """Calculate the volume of the belt.
        
        The belt volume is calculated assuming it to be a torus with
        radius r0_cm, radial width dr_r * r0_cm, and vertical thickness
        determined by the maximum inclination.

        Returns:
            float: Belt volume (cm³)
        """
        return 4 * pi * self.r0_cm**3 * self.dr_r * self.inc_max_rad

    def calculate_properties(self, prtl: 'Particles', star: 'Star'):
        """Calculate all derived belt properties. This method should be called
        after initialization but before the belt is used to create a disk.
        Usually, this method is called by the factory method Disk.make_disk().

        Args:
            prtl: Particles object containing particle parameters
            star: Star object containing stellar parameters

        Order of calculations:
        1. Collisional velocity
        2. Critical impactor size
        3. Size distribution
        4. Lifetimes and ratios
        """
        # Calculate properties in required order
        self.calculate_coll_velocity(star.gm_cgs)
        self.calculate_critical_impactor_size(prtl.Q_D)
        self.calculate_size_distribution_both_regimes(prtl, star)
        self.calculate_ratio_lifetimes_prdrag_to_coll(prtl, star)

    def calculate_coll_velocity(self, star_gm_cgs):
        """Calculate the characteristic collisional velocity in the belt.
        Corresponding to Wyatt & Dent (2002), i.e., mean collisional velocity is calculated
        from the Keplerian velocity and RMS of mean orbital eccentricities and inclinations.

        Args:
            star_gm_cgs (float): Standard gravitational parameter of star (CGS)

        Sets:
            self.vel_coll (float): Mean collisional velocity (cm/s)
        """
        vel_kepler = np.sqrt(star_gm_cgs / self.r0_cm)  # Keplerian velocity in the belt
        self.vel_coll = vel_kepler * np.sqrt(1.25*self.avg_ecc**2 + self.inc_max_rad**2) # Wyatt & Dent 2002

    def calculate_critical_impactor_size(self, Q_D):
        """Calculate critical impactor size ratio for catastrophic disruption.

        Corresponding to Rigley & Wyatt (2020), Eq. (4).

        Args:
            Q_D (float): Critical specific energy "Q_D star" (erg/g)

        Sets:
            self.X_C (float): Critical impactor size ratio (-)
        """
        self.X_C = (2 * Q_D / self.vel_coll**2)**(1/3)

    def calculate_size_distribution_both_regimes(self, prtl, star):
        """Calculate and set the size distribution within the belt across the 
        collisional and PR-drag regimes.

        Args:
            prtl (Particles): Object holding the model input particle parameters
            star (Star): Object holding the model input star parameters

        Sets:
            self.alpha (float): Size distribution slope (-)
            self.K_norm (float): Normalization factor for collisional regime
            self.n_D (np.ndarray): Size distribution (-)
            self.D_pr (float): PR-drag regime turnover size (cm)
            self.K_norm_pr (float): Normalization factor for PR-drag regime
        """
        # Size distribution in coll regime
        self.alpha = (7 - prtl.a/3) / (2 - prtl.a/3)  # size distribution slope RW20 Eq. (14)
        self.K_norm = Belt.calculate_size_dist_norm(self.alpha, prtl.matrl.density, prtl.diam_max, self.m_dust_g)
        self.n_D = Belt.calculate_power_law_distribution(prtl.diams, norm_factor=self.K_norm, slope=self.alpha)

        # Find D_pr, the regime boundary where eta_0 = 1
        self.D_pr = Belt.calculate_turnover_size_sizedist(star.gm_cgs, star.lum_cgs, self.alpha, 
                                                          prtl.diam_max, prtl.Qa, prtl.a, self.m_dust_g, 
                                                          self.inc_max_rad, self.dr_r, self.r0_cm)

        # Size distribution in pr regime
        self.K_norm_pr = Belt.calculate_size_dist_norm_prdrag(self.K_norm, self.D_pr, self.alpha, prtl.alpha_r)
        n_D_pr = Belt.calculate_power_law_distribution(prtl.diams, norm_factor=self.K_norm_pr, slope=prtl.alpha_r - 1)

        # Join both regimes
        self.n_D[prtl.diams <= self.D_pr] = n_D_pr[prtl.diams <= self.D_pr]

    def calculate_ratio_lifetimes_prdrag_to_coll(self, prtl, star):
        """Calculate and set the ratios of the PR-drag and collisional timescale across both, 
        the PR-drag and the collisional regime.

        Args:
            prtl (Particles): Object holding the model input particle parameters.
            star (Star): Object holding the model input paramters of the star.
        """
        # PR drag timescales
        self.t_PR = Belt.calculate_lifetime_prdrag(self.r0_cm, star.gm_cgs, prtl.betas)

        # Collisional timescales in coll regime
        t_coll_coll = Belt.calculate_lifetime_collisions(self.K_norm, self.alpha, prtl.diams, self.vol, self.vel_coll, self.X_C)
        
        # Collisional timescales in PR regime
        t_coll_pr = Belt.calculate_lifetime_collisions_prdrag(self.K_norm_pr, prtl.alpha_r, prtl.diams, 
                                                     self.vol, self.vel_coll, self.X_C)
        
        # Find D_pr_eff where the two regimes intersect.
        # This is equivalent to where X_CD = D_pr, with an extra factor O(1)
        self.D_pr_eff = Belt.calculate_turnover_size_collisions(self.vel_coll, self.alpha, 
                                                      prtl.alpha_r, prtl.Qa, prtl.a, self.D_pr)
        
        t_coll = np.where(prtl.diams <= self.D_pr_eff,
                          t_coll_pr,                # PR drag regime
                          t_coll_coll)              # Collisional regime
        
        self.t_coll_k = t_coll * prtl.k_factor      # include fudge factor
        self.eta_0 = self.t_PR / self.t_coll_k

    @staticmethod
    def calculate_turnover_size_sizedist(star_gm_cgs, star_lum_cgs, alpha, diam_max, 
            Qa, a, m_dust_g, inc_max_rad, dr_r, r0_cm):
        """Calculate and set the grain size for the transition from collisional to 
        PR-drag-dominated regimes for the size distribution.
        At this size, D_pr, the collisional lifetime is equal to the PR-drag lifetime. 
        Accordingly D_pr is found analytically by solving the equation
        t_coll(D,belt_r0) = t_PR(D,belt_r0) for D.
        
        Args:
            star (Star): Object holding the model input paramters of the star.
            diam_max (float): Maximum particle size (cm)
            Qa (float): Normalization parameter of "Q_D star" (erg/g)
            a (float): Slope parameter of "Q_D star" (-)

        Returns:
            float: Turnover grain diamter, D_pr (cm)
        """
        return ((alpha - 1) / (4 - alpha) * pow(2, 1/3*(5 + alpha)) / c_cgs**2 * 
                pow(star_gm_cgs, -(1 + 2 * alpha)/6) * star_lum_cgs / m_dust_g *
                pow(diam_max, 4 - alpha) * pow(Qa, (alpha - 1)/3) * 
                pow(inc_max_rad, -2/3*(alpha - 1)) * dr_r * 
                pow(r0_cm, 7/6 + alpha/3))**(1/(4 - alpha - a/3*(1 - alpha)))

    @staticmethod
    def calculate_turnover_size_collisions(belt_vel_coll, alpha, alpha_r, Qa, a, D_pr):
        """Calculate the grain size for the transition from collisional to PR-drag-dominated regimes
        of the collisional lifetime. Corresponding to Rigley 2022 (PhD thesis), Eq. (2.19). 
        CAUTION: Rigley & Wyatt (2020), Eq. (19) is errorneous.

        Args:
            belt (Belt): Object holding the paramters of the belt.
            star (Star): Object holding the paramters of the star.
            alpha (float): Slope of size distribution (-)
            Qa (float): Normalization parameter of "Q_D star" (erg/g)
            a (float): Slope parameter of "Q_D star" (-)

        Returns:
            float: Turnover grain diamter, D_pr_eff (cm)
        """
        return (pow((alpha_r - 2)/(alpha - 1), 3/((3 - a) * (1 + alpha - alpha_r)))
             * pow(belt_vel_coll**2/(2*Qa), 1/(3 - a)) * pow(D_pr, 3/(3 - a)))

    @staticmethod
    def calculate_power_law_distribution(diams, norm_factor, slope):
        """Calculates the number of particles at different particle size according to a power law.
        Corresponds to Rigley & Wyatt (2020), Eq. (7) and (15).

        Args:
            diams (float): Particle diamteres (cm)
            norm_factor (float): Normalization factor for size distribution (unit: TODO)
            slope (float): Size distribution slope (-)
        
        Returns:
            float: Particle numbers, n_D (-)
        """
        return norm_factor * pow(diams, -slope)

    @staticmethod
    def calculate_lifetime_collisions_prdrag(K_norm_pr, alpha_r, diams, belt_vol, belt_vel_coll, X_C):
        """Calculates the collisions timescales in the PR-regime.
        Corresponds to Rigley & Wyatt (2020), Eqs. (16) & (17).

        Args:
            K_norm_pr (float): Normalization factor for size distribution in PR drag regime (unit: TODO)
            alpha_r (float): Slope of redistribution function (-)
            belt_vol (float): Volume of planetesimal belt (cm³)
            belt_vel_coll (float): Collisional velocity (cm/s)
            X_C (float): Critical impactor size ratio (-)

        Returns:
            float: Collisional timescales, t_coll_pr (yr)
        """
        return (4 * (alpha_r - 2) * belt_vol / (pi * K_norm_pr * belt_vel_coll) * 
                pow(X_C, alpha_r - 2) * pow(diams, alpha_r - 4)) / year_seconds

    @staticmethod
    def calculate_lifetime_collisions(K_norm, alpha, diams, belt_vol, belt_vel_coll, X_C):
        """Calculate collisional timescales of dust grains.

        Args:
            K_norm (float): Size distribution normalization factor
            alpha (float): Size distribution slope (-)
            diams (np.ndarray): Particle diameters (cm)
            belt_vol (float): Belt volume (cm³)
            belt_vel_coll (float): Collisional velocity (cm/s)
            X_C (float): Critical impactor size ratio (-)

        Returns:
            np.ndarray: Collisional timescales, t_coll (yr)
        """
        return (4 * (alpha - 1) * belt_vol / (pi * K_norm * belt_vel_coll) *
            pow(X_C, alpha - 1) * pow(diams, alpha - 3)) / year_seconds

    @staticmethod
    def calculate_lifetime_prdrag(orbit_radius, star_gm_cgs, betas):
        """Calculate the Poynting-Robertson drag lifetime from a circular orbit for a set of beta factors.
        This function returns the drag lifetime for each provided beta factor, 
        assuming all particles are at the same initial orbit.

        Args:
            orbit_radius (float): semi-major axis of the initial circular orbit (cm)
            star_gm_cgs (float): standard gravitational parameter of the star (CGS units)
            betas (float): particle beta factors (-)

        Returns:
            float: The drag lifetimes per beta value, t_PR (yr)
        """
        return (c_cgs * orbit_radius**2 / (4 * star_gm_cgs * betas)) / year_seconds

    @staticmethod
    def calculate_size_dist_norm_prdrag(K_norm, D_pr, alpha, alpha_r):
        """Calculate the normalization factor for the size distribution in the PR drag regime.
        Corresponding to Rigley & Wyatt (2020), Eq. (16).

        Args:
            K_norm (float): Normalization factor for size distribution in the collisional regime (unit: TODO)
            D_pr (float): Turnover size between collisional and PR drag regime (cm)
            alpha (float): Size distribution slope (-)
            alpha_r (float): Slope of redistribution function (-)

        Returns:
            float: Normalization factor for size distribution in PR drag regime, K_norm_pr (unit: TODO)
        """
        return K_norm * pow(D_pr, alpha_r - alpha - 1)

    @staticmethod
    def calculate_size_dist_norm(alpha, density, diam_max, m_dust_g):
        """Calculate the normalization factor for the size distribution in the collisional regime.
        Corresponding to Rigley & Wyatt (2020), Eq. (10).

        Args:
            alpha (float): Size distribution slope (-)
            density (float): Particle bulk density (g/cm³)
            diam_max (float): Maximum particle size (cm)
            m_dust (float): Total dust mass (g)

        Returns:
            float: Normalization factor for size distribution in coll. regime, K_norm (unit: TODO)
        """
        return 6 * (4 - alpha) / (pi * density) * pow(diam_max, alpha - 4) * m_dust_g

    def estimate_total_mass_from_input_rate(self, prtl, m_dust_input_earths=None, m_dust_input_g=None):
        """Calculates the total dust mass based on the mass input rate.
        Corresponding to Rigley & Wyatt (2020), Eq. (9), (10), and (27).
        """
        if len(self.X_C)==0:
                belt.calculate_critical_impactor_size(prtl.Q_D)
            
        K_norm_over_mass = 6 * (4 - self.alpha) / pi / prtl.matrl.density * pow(prtl.diam_max, self.alpha - 4)

        t_coll_max_times_mass = (4 * (self.alpha - 1) * self.vol / (K_norm_over_mass * pi * self.vel_coll)
                                * self.X_C[-1]**(self.alpha - 1) * prtl.diam_max**(self.alpha -3) / year_seconds)

        t_coll_k_max_times_mass = t_coll_max_times_mass * prtl.k_factor[-1] # include fudge factor                                
        
        if m_dust_input_g:
            self.m_dust_g = np.sqrt(m_dust_input_g * t_coll_k_max_times_mass)
            self.m_dust_earths = (self.m_dust_g * u.g).to(u.M_earth).value
        elif m_dust_input_earths:
            m_dust_input_g = (m_dust_input_earths * const.M_earth).to(u.g).value
            self.m_dust_g = np.sqrt(m_dust_input_g * t_coll_k_max_times_mass)
            self.m_dust_earths = (self.m_dust_g * u.g).to(u.M_earth).value
        else:
            raise ValueError("No mass input rate passed. Pass either 'm_dust_input_earths' or 'm_dust_input_g'.")

    def estimate_input_rate_from_total_mass(self):
        """Calculates the total dust mass based on the mass input rate.
        Corresponding to Rigley & Wyatt (2020), Eq. (27).
        """
        self.m_dust_inputrate_g = self.m_dust_g / self.t_coll_k[-1]
        self.m_dust_inputrate_earths = (self.m_dust_inputrate_g * u.g).to(u.M_earth).value