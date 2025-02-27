import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.interpolate import interp1d
import astropy.units as u


class Disk:
    """Disk object holding computed results
    """
    # Class attributes
    tau_Dr: np.ndarray  # 2D distribution of optical depth
    tau_r: np.ndarray   # 1D radial optical depth profile
    sig_dlogD: np.ndarray  # 2D distribution of cross-sectional area

    @classmethod
    def make_disk(cls, 
                  belt: 'Belt',
                  prtl: 'Particles',
                  rbin: 'RadialBins',
                  star: 'Star',
                  dlog: bool = True,
                  verbose: bool = True) -> 'Disk':
        """Create a Disk instance from constituent class objects.
        
        Args:
            belt: Belt object containing belt parameters
            prtl: Particles object containing particle parameters
            rbin: RadialBins object containing radial binning parameters
            star: Star object containing stellar parameters
            dlog: Whether to use logarithmic binning
            verbose: Whether to print diagnostic information
            
        Returns:
            Disk: A new Disk instance with computed properties
        """
        # Initialize required components
        prtl.calculate_k_factor_tcoll()
        prtl.interpolate_temperatures(rbin)
        prtl.bnus = prtl.calculate_spectral_radiance_bb(prtl.wavs, prtl.temps)

        # Calculate belt properties
        belt.calculate_properties(prtl, star)

        if verbose:
            print("Material density: " + f"{prtl.matrl.density:.2f}" + " g/cm3\n " + prtl.matrl.info)
            print("Belt size distribution slope: alpha = " + f"{belt.alpha:.2f}")
            print("mean X_C = " + f"{np.mean(belt.X_C):.2f}")
            print('D_pr     = %.2f µm\nD_pr_eff = %.2f µm\nD_bl     = %.2f µm' %
                (belt.D_pr*1e4, belt.D_pr_eff*1e4, prtl.diams_blow[1]*1e4))

        # Create and configure disk
        disk = cls()
        disk.optical_depth(belt, prtl, rbin, dlog=dlog)
        disk.cross_sect_area(rbin)

        rbin.calculate_solid_angle_of_annuli(star.dist_au)
        disk.compute_sed(prtl=prtl, rbin=rbin)

        return disk

    def optical_depth(self, belt, prtl, rbin, dlog=False):
        """Calculate the two-dimensional optical depth distribution of the disc.
        Corresponding to Rigley & Wyatt (2020), Eqs. (22) & (23).

        Args:
            belt (Belt): Object holding the model input parameters of the belt
            prtl (Particles): Object holding the model input particle parameters
            rbin (RadialBins): Object holding the radial binning parameters
            dlog (bool): Whether to use logarithmic binning

        Sets:
            self.tau_Dr: 2D distribution of optical depth over particle size and radial bins,
            either as dtau/dlogD or dtau/dD depending on dlog flag. (Shape: nD x nr). (Unit: - )
            self.tau_r: 1D radial optical depth profile
        """
        # Calculate tau_0 (optical depth in belt) according to RW20, Eq. (22)
        tau_0 = belt.n_D * prtl.diams**2 / (8 * belt.r0_cm**2 * belt.dr_r)
        if dlog:
            tau_0 = tau_0 * prtl.diams * np.log(10)

        # Calculate tau_Dr for all radii interior to the belt, according to RW20, Eq. (23)
        tau_Dr = np.zeros((prtl.n_diams, rbin.num))
        for iD in range(prtl.n_diams):
            for ir in range(len(rbin.mids_cm[rbin.mids_cm <= belt.r0_cm])):
                tau_Dr[iD, ir] = tau_0[iD] / (1 + 4 * belt.eta_0[iD] * (1 - np.sqrt(rbin.mids_cm[ir] / belt.r0_cm)))

        # Within the belt, tau_Dr is just tau_0. Exterior to the belt, tau_Dr is zero.
        for ir in range(rbin.num):
            if (rbin.mids_cm[ir] >= belt.r0_cm) & (rbin.mids_cm[ir] <= belt.r_out_cm):
                tau_Dr[:, ir] = tau_0

        # Set tau_Dr to zero for sizes below blowout size
        tau_Dr = np.where((prtl.diams < prtl.diam_min)[:, np.newaxis], 0, tau_Dr)

        # Integrate over log bins of D to create 1D (radial) optical depth profile
        tau_r = scipy.integrate.simpson(tau_Dr, x=np.log10(prtl.diams), axis=0)
        
        self.tau_Dr = tau_Dr
        self.tau_r = tau_r

    def cross_sect_area(self, rbin):
        """
        Calculate and set the two-dimensional distribution of cross-sectional area
        in AU^2 for each (D,r) bin. 
        
        Args:
            rbin (RadialBins): Object holding the radial binning parameters
        """
        # Calculate total cross-sectional area (CSA) in AU^2 for each (D,r) bin
        self.sig_dlogD = self.tau_Dr * rbin.annuli_area[np.newaxis, :]

    def compute_sed(self, prtl, rbin, wavs=None):
        """Calculate the spectral energy distribution (SED) of the disk.

        Args:
            prtl (Particles): Object containing particle parameters.
            rbin (RadialBins): Object containing radial binning parameters.
            wavs (list, optional): Wavelengths at which to calculate the SED (microns).

        Raises:
            ValueError: If rbin.annuli_solidang is None
        """
        if rbin.annuli_solidang is None:
            raise ValueError("rbin.annuli_solidang is None. Solid angles must be computed before computing SED.")
        
        if wavs is None:
            self.sed_wavs = prtl.wavs
        
        n_wavs = len(self.sed_wavs)
        self.sed_flux = np.zeros(n_wavs) * u.Jy
        self.sed_flux_2d = np.zeros([n_wavs, rbin.num])  * u.Jy

        for i_w, wav in enumerate(self.sed_wavs):
            S_nu = self.calculate_surface_brightness(wav=wav, prtl=prtl)
            F_nu = S_nu * rbin.annuli_solidang
            self.sed_flux[i_w] = np.sum(F_nu)
            self.sed_flux_2d[i_w, :] = F_nu

    def calculate_surface_brightness(self, wav, prtl, interp=True):
        # integrate Qabs * Bnu(T(D,r)) * CSA to get radial flux profiles at wavelength nu in Jy
        # this gives the total flux (Jy) from each radial bin
        # RW20, Eq 34 (I verified that dstar_au**-2 is the same)

        # Add the capability to use Qabs2d / particles with changing composition...

        if interp and not np.isclose(prtl.wavs, wav, atol=1e-3).any():
            # Interpolate Qabs and bnus for the given wavelength
            Qabs_interp = interp1d(prtl.wavs, prtl.Qabs, axis=1, kind='linear', fill_value="extrapolate")
            bnus_interp = interp1d(prtl.wavs, prtl.bnus, axis=2, kind='linear', fill_value="extrapolate")
            
            # Evaluate at the specified wavelength
            Qabs_at_wav = Qabs_interp(wav)
            bnus_at_wav = bnus_interp(wav)
        else:
            # Find the nearest wavelength index
            i_nu = np.abs(prtl.wavs - wav).argmin()
            Qabs_at_wav = prtl.Qabs[:, i_nu]
            bnus_at_wav = prtl.bnus[:, :, i_nu]

        # Calculate F_nu
        S_nu_per_size = Qabs_at_wav[None, :] * bnus_at_wav * self.tau_Dr.T
        S_nu = scipy.integrate.simpson(S_nu_per_size, x=np.log10(prtl.diams), axis=1) * u.Jy / u.sr

        return S_nu

    def calculate_zodi_level(self, star_lum_suns, rbin):
        """Calculate the zodi level of the disk relative to the solar system's zodiacal cloud,
        according to Kennedy et al. (2015).
        
        The zodi level is determined by comparing the disk's optical depth at the equivalent
        Earth-Sun separation (scaled by stellar luminosity) to the solar system's zodiacal cloud
        optical depth at 1 AU.
        
        Args:
            star (Star): Star object containing stellar parameters
            rbin (RadialBins): Object containing radial binning parameters
            
        Returns:
            float: Zodi level relative to solar system (1.0 = solar system level)
        """
        # Calculate the Earth-equivalent insolation distance for this star
        r0_zodi_au = np.sqrt(star_lum_suns)
        
        # Solar system zodiacal cloud optical depth at 1 AU
        surf_dens_1zodi_at_r0 = 7.12e-8
        
        # Interpolate the disk's optical depth to get value at r0_zodi_au
        surf_dens_model_at_r0 = np.interp(x=r0_zodi_au, xp=rbin.mids, fp=self.tau_r, 
                                          left=0.0, right=0.0)
        
        surf_dens_1zodi_at_r0 = 7.12e-8
        
        # Calculate zodi level
        zodi_level = surf_dens_model_at_r0 / surf_dens_1zodi_at_r0
        
        return zodi_level

    def plot_optical_depth_distribution(self, prtl_diams: np.ndarray, rbin_mids: np.ndarray, 
                                   r_lims=None, tau_Dr_lims=None, tau_Dr_step=1, 
                                   tau_Dr_nlevels=30, cbextend='min', cmap='viridis', 
                                   tick_contours=False):
        """Plot optical depth distribution of the disk.
        
        Args:
            prtl_diams (np.ndarray): Particle diameters (cm)
            rbin_mids (np.ndarray): Radial bin centers (au)
            r_lims (tuple, optional): (min, max) radius to plot (au)
            tau_Dr_lims (tuple, optional): (min, max) tau values to plot
            tau_Dr_step (float): Step size for tau contours
            tau_Dr_nlevels (int): Number of contour levels
            cbextend (str): Colorbar extend parameter
            cmap (str): Matplotlib colormap name
            tick_contours (bool): Whether to show contour ticks

        Returns:
            tuple: (figure, (ax1, ax2)) matplotlib figure and axes objects
        """
        grid_kw = {'height_ratios':[1,3]}
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, gridspec_kw=grid_kw)
        
        # tau_r plot
        ax1.loglog(rbin_mids, self.tau_r)
        ax1.set_ylabel(r'$\tau$')
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.yaxis.set_ticks_position('right')
        ax1.tick_params(axis='both', direction='in', which='major', bottom=True, left=True, right=True, top=True, length=4)
        ax1.tick_params(axis='both', direction='in', which='minor', bottom=True, left=True, right=True, top=True, length=3)
        ax1.yaxis.set_label_position('right')
        tau_r_lims = (np.floor(np.min(np.log10(np.where(self.tau_r > 0, self.tau_r, 1e99)))),
                      np.ceil(np.max(np.log10(np.where(self.tau_r > 0, self.tau_r, 1e-99)))))
        ax1.set_ylim(10**tau_r_lims[0], 10**tau_r_lims[1])
        if r_lims is not None:
            ax1.set_xlim(r_lims[0], r_lims[1])

        # tau_Dr plot
        if tau_Dr_lims is None:
            tau_Dr_lims = (np.floor(np.min(np.log10(np.where(self.tau_Dr > 0, self.tau_Dr, 1e99)))),
                          np.ceil(np.max(np.log10(np.where(self.tau_Dr > 0, self.tau_Dr, 1e-99)))))
        else:
            if tau_Dr_lims[0] == -float('inf'):
                tau_Dr_lims[0] = np.min(np.log10(np.where(self.tau_Dr > 0, self.tau_Dr, 1e99)))
            if tau_Dr_lims[1] == float('inf'):
                tau_Dr_lims[1] = np.max(np.log10(np.where(self.tau_Dr > 0, self.tau_Dr, 1e-99)))

        ticks_reversed = np.arange(np.floor(tau_Dr_lims[1]), np.floor(tau_Dr_lims[0]) - tau_Dr_step * .6, -tau_Dr_step)
        ticks = ticks_reversed[::-1]
        cblabels = [r'$10^{{{:.0f}}}$'.format(t) for t in ticks]
        levels = np.linspace(ticks[0], tau_Dr_lims[1], num=tau_Dr_nlevels)
        
        rads, diams = np.meshgrid(rbin_mids, prtl_diams)
        cp = ax2.contourf(rads, diams*1e4, np.log10(np.where(self.tau_Dr > 0, self.tau_Dr, 1e-300)), 
                         levels=levels, extend=cbextend, cmap=cmap)
        cp.set_edgecolor('face')

        cb = fig.colorbar(cp, ax=ax2, ticks=ticks, pad=-.1, aspect=16, shrink=.9)
        cb.ax.set_yticklabels(cblabels)
        cb.set_label(r'$d\tau \,/\, d \mathrm{log_{10}} \, D$')

        if tick_contours:
            contour_lines = ax2.contour(rads, diams * 1e4, np.log10(np.where(self.tau_Dr > 0, self.tau_Dr, 1e-300)),
                                      levels=ticks[1::], colors='white', linewidths=0.5, alpha=.5)
            cb.add_lines(contour_lines)
        
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_xlabel('Radial distance (AU)')
        ax2.set_ylabel(r'$D$ (µm)')
        ax2.tick_params(axis='both', direction='in', which='major', bottom=True, left=True, right=True, top=True, length=4)
        ax2.tick_params(axis='both', direction='in', which='minor', bottom=True, left=True, right=True, top=True, length=3)
        
        fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

        return fig, (ax1, ax2)
