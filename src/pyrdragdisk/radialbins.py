import numpy as np
import astropy.units as u
        
class RadialBins:
    """Configuration parameters for radial grid of the model.
    
    Attributes:
        r_min (float): Minimum radius (au)
        r_max (float): Maximum radius (au)
        num (int): Number of radial bins
        mids (np.ndarray): Center radii of bins (au)
        edges (np.ndarray): Edge radii of bins (au)
        widths (np.ndarray): Width of each bin (au)
        mids_cm (np.ndarray): Center radii in cm
        annuli_area (np.ndarray): Area of each annulus (au²)
        annuli_solidang_sr (np.ndarray): Solid angle of each annulus (sr)
        annuli_solidang_sqas (np.ndarray): Solid angle of each annulus (arcsec²)
    """
    def __init__(self, r_min_au: float, r_max_au: float, n_bin: int, spacing: str = "lin") -> None:
        if r_min_au <= 0:
            raise ValueError(f"r_min must be positive (got {r_min})")
        if r_max_au <= r_min_au:
            raise ValueError(f"r_max ({r_max}) must be greater than r_min ({r_min})")
        if n_bin < 2:
            raise ValueError(f"Number of bins must be at least 2 (got {n_bin})")

        # Initialize basic parameters
        self.r_min = r_min_au
        self.r_max = r_max_au
        self.num = n_bin
        
        # Initialize arrays that will be set by make_*_bins methods
        self.mids: np.ndarray = None                # Center radii (au)
        self.edges: np.ndarray = None               # Edge radii (au)
        self.widths: np.ndarray = None              # Bin widths (au)
        self.mids_cm: np.ndarray = None             # Center radii (cm)
        self.annuli_area: np.ndarray = None         # Annuli areas (au²)
        self.annuli_solidang: np.ndarray = None     # Solid angles (u.Unit('sr'))

        # Create bins based on spacing type
        if spacing == "lin" or spacing == "linear":
            self._make_linear_bins()
        elif spacing == "log":
            self._make_log_bins()
        elif spacing == "sqrt":
            self._make_sqrt_bins()
        else:
            raise ValueError(f"Unknown spacing method ({spacing}). Use 'lin', 'log', or 'sqrt'.")

        if self.edges[0] < 0:
            raise ValueError(f"Lower edge of first bin is negative ({self.edges[0]:.2f})")

        # Calculate derived quantities
        self.widths = np.diff(self.edges)
        self.annuli_area = 2 * np.pi * self.mids * self.widths      # au²
        self.mids_cm = self.mids * u.au.to(u.cm)

    def _make_linear_bins(self) -> None:
        """Create linearly spaced bins."""
        width = (self.r_max - self.r_min) / (self.num - 1)
        self.mids = np.linspace(self.r_min, self.r_max, num=self.num)
        self.edges = np.append([self.mids - width/2], [self.mids[-1] + width/2])
    
    def _make_log_bins(self) -> None:
        """Create logarithmically spaced bins."""
        self.mids = np.logspace(np.log10(self.r_min), np.log10(self.r_max), num=self.num)
        # Calculate edges as geometric mean between centres
        edges_between = np.sqrt(self.mids[:-1] * self.mids[1:])
        first_edge = self.mids[0] / np.sqrt(self.mids[1] / self.mids[0])
        last_edge = self.mids[-1] * np.sqrt(self.mids[-1] / self.mids[-2])
        self.edges = np.concatenate(([first_edge], edges_between, [last_edge]))

    def _make_sqrt_bins(self) -> None:
        """Create square-root spaced bins."""
        sqrt_min = np.sqrt(self.r_min)
        sqrt_max = np.sqrt(self.r_max)
        sqrt_mids = np.linspace(sqrt_min, sqrt_max, num=self.num)
        
        # Calculate bin edges on sqrt scale
        sqrt_edges_between = np.sqrt((sqrt_mids[:-1]**2 + sqrt_mids[1:]**2) / 2)
        sqrt_first_edge = np.sqrt(sqrt_mids[0]**2 - (sqrt_mids[1]**2 - sqrt_mids[0]**2) / 2)
        sqrt_last_edge = np.sqrt(sqrt_mids[-1]**2 + (sqrt_mids[-1]**2 - sqrt_mids[-2]**2) / 2)
        
        # Convert back to original scale
        self.mids = sqrt_mids**2
        edges_between = sqrt_edges_between**2
        first_edge = sqrt_first_edge**2
        last_edge = sqrt_last_edge**2
        self.edges = np.concatenate(([first_edge], edges_between, [last_edge]))

    def calculate_solid_angle_of_annuli(self, star_dist_au: float) -> None:
        """Calculate solid angles of each annulus.
        
        Args:
            star_dist_au (float): Distance to star (au)
        """
        self.annuli_solidang = self.annuli_area / star_dist_au**2 * u.steradian