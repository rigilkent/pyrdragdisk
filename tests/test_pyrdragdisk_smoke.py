import numpy as np
import astrodust_optprops as opt
import astropy.units as u
import pyrdragdisk as pyr
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

tests_dir = Path(__file__).parent
optmod_file = tests_dir / 'fomalhaut_optmod.pkl'

def test_pyrdragdisk_smoke():
    # Get optical properties generated with optprops
    if optmod_file.exists():
        # Load from file
        optmod = opt.OpticalModel.load(optmod_file)
    else:
        # Run optprops
        star = opt.Star(name='Fomalhaut', lum_suns=16.6, mass_suns=1.92, temp=8590)
        matrl = opt.Material(qsil=.4, qice=1.0, mpor=.7)
        diams = np.logspace(.5, 5, 55)
        dists = np.logspace(-.5, 2.5, 67)
        wavs = np.logspace(.5, 4, 195)
        prtl = opt.Particles(diams=diams, wavs=wavs, matrl=matrl, dists=dists, suppress_mie_resonance=True)
        prtl.calculate_all(star)
        optmod = opt.OpticalModel(star=star, prtl=prtl)
        optmod.save(optmod_file)

    # ---------------- Run pyrdragdisk to generate disk model -----------------
    # Create parameter object for the star
    star = pyr.Star(dist_pc=7.7, optprops_star=optmod.star)

    # Create parameter object for the belt
    belt = pyr.Belt(r0_au=120,                      # inner radius (au)
                    dr_r=0.385,                     # width relative to r0
                    inc_max_deg=2,                  # max inclination (deg)
                    m_dust_earths=0.015)            # total dust mass (Earth masses)

    # Create parameter object for radial grid
    rbin = pyr.RadialBins(r_min_au=0.5,              # min radius (au)
                        r_max_au=200,                # max radius (au)
                        n_bin=200,                   # number of bins
                        spacing='sqrt')              # bin spacing ('lin', 'log', or 'sqrt')

    # Create parameter object for the particles
    prtl = pyr.Particles(optprops_prtl=optmod.prtl,     # object holding optical properties (optprops.OpticalModel)
                        qd_norm_cgs=3e6,               # Strength law scaling parameter (erg/g)
                        qd_slope=0.0,                  # Strength law slope parameter
                        diam_max_cm=1)                 # Maximum particle size to consider (cm)


    # --- Make disk ---
    disk = pyr.Disk.make_disk(belt=belt, prtl=prtl, rbin=rbin, star=star)

    # Calculate surface brightness at MIRI wavelengths
    S_15 = disk.calculate_surface_brightness(wav=15.5, prtl=prtl)
    S_23 = disk.calculate_surface_brightness(wav=23.0, prtl=prtl)
    S_25 = disk.calculate_surface_brightness(wav=25.5, prtl=prtl)
    sed_no_pl = disk.sed_flux

    # Apply planet depletion
    disk.apply_bonsor18_inner_depletion(
        a_pl=50 * u.au,
        M_pl=1 * u.M_jup,
        star=star,
        rbin=rbin,
        prtl=prtl
    )
    S_15_pl = disk.calculate_surface_brightness(wav=15.5, prtl=prtl)
    S_23_pl = disk.calculate_surface_brightness(wav=23.0, prtl=prtl)
    S_25_pl = disk.calculate_surface_brightness(wav=25.5, prtl=prtl)


    # Plot radial surface brightness profiles
    fig, ax = plt.subplots()
    ax.loglog(rbin.mids, S_25_pl.to(u.MJy/u.sr), 'r--')
    ax.loglog(rbin.mids, S_23_pl.to(u.MJy/u.sr), 'g--')
    ax.loglog(rbin.mids, S_15_pl.to(u.MJy/u.sr), 'b--')
    ax.loglog(rbin.mids, S_25.to(u.MJy/u.sr), 'r-', label='25.5 µm')
    ax.loglog(rbin.mids, S_23.to(u.MJy/u.sr), 'g-', label='23.0 µm')
    ax.loglog(rbin.mids, S_15.to(u.MJy/u.sr), 'b-', label='15.5 µm')
    ax.set_xlim(3, 300)
    ax.set_ylim(.1, 1e3)
    ax.set_xlabel('Radial distance (au)')   
    ax.set_ylabel(r'Surface brightness (MJy sr$^{-1}$)')
    ax.legend()
    fig.savefig(tests_dir / 'rad_prof.png')


    # Plot SED
    fig, ax = plt.subplots()
    ax.loglog(disk.sed_wavs, sed_no_pl, 'k-', label='No planet')
    ax.loglog(disk.sed_wavs, disk.sed_flux, 'k--', label='With planet')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Flux (Jy)')
    ax.set_xlim(5, 2000)
    ax.set_ylim(1e-3, 100)
    fig.savefig(tests_dir / 'SED.png')


# Run file as script to execute all test defined in this file
if __name__ == '__main__':
    pytest.main([tests_dir / 'test_pyrdragdisk_smoke.py', '--verbose'])



# If want to implement more tests, consider using fixtures:
# https://docs.pytest.org/en/stable/explanation/fixtures.html