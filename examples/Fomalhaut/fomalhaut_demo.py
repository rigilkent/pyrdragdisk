import numpy as np
import matplotlib.pyplot as plt
import astrodust_optprops as opt
import astropy.units as u
import pyrdragdisk as pyr
from pathlib import Path

script_dir = Path(__file__).parent
optmod_file = script_dir / 'fomalhaut_optmod.pkl'
savefig_dict = {'bbox_inches': 'tight', 'pad_inches': 0.1, 'dpi': 300}


# -------------- Run optprops to generate optical properties --------------
star = opt.Star(name='Fomalhaut', lum_suns=16.6, mass_suns=1.92, temp=8590)
matrl = opt.Material(qsil=.4, qice=1.0, mpor=.7)
diams = np.logspace(.5, 5, 55)
dists = np.logspace(-.5, 2.5, 67)
wavs = np.logspace(.5, 4, 195)
prtl = opt.Particles(diams=diams, wavs=wavs, matrl=matrl, dists=dists, suppress_mie_resonance=True)
prtl.calculate_all(star)
optmod = opt.OpticalModel(star=star, prtl=prtl)
optmod.save(optmod_file)

# Load star and particle optical properties generated with optprops
# optmod = opt.OpticalModel.load(optmod_file)

ax = optmod.prtl.plot_Qabs(diams=np.logspace(1, 3, 5))
ax.figure.savefig(script_dir / 'Qabs.png', **savefig_dict)

# ---------------- Run pyrdragdisk to generate disk model -----------------
# Create parameter object for the star
star = pyr.Star(dist_pc=7.7, optprops_star=optmod.star)

# Create parameter object for the belt
belt = pyr.Belt(r0_au=120,                      # inner radius (au)
                dr_r=0.385,                     # width relative to r0
                inc_max_deg=2,                # max inclination (deg)
                m_dust_earths=0.015)            # total dust mass (Earth masses)

# Create parameter object for radial grid
rbin = pyr.RadialBins(r_min_au=0.5,                # min radius (au)
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

# Plot radial surface brightness profiles
fig, ax = plt.subplots()
ax.loglog(rbin.mids, S_25.to(u.MJy/u.sr), 'r-', label='25.5 µm')
ax.loglog(rbin.mids, S_23.to(u.MJy/u.sr), 'g-', label='23.0 µm')
ax.loglog(rbin.mids, S_15.to(u.MJy/u.sr), 'b-', label='15.5 µm')
ax.set_xlim(3, 300)
ax.set_ylim(.1, 1e3)
ax.set_xlabel('Radial distance (au)')   
ax.set_ylabel(r'Surface brightness (MJy sr$^{-1}$)')
ax.legend()
fig.savefig(script_dir / 'rad_prof.png', **savefig_dict)

#  Plot optical depth distribution
fig, _ = disk.plot_optical_depth_distribution(prtl.diams, rbin.mids,
                r_lims=(2, 200), tau_Dr_lims=[-10, float('inf')], 
                tau_Dr_step=2, tau_Dr_nlevels=27, 
                cmap='viridis', tick_contours=False)
fig.savefig(script_dir / 'tau.png', **savefig_dict)


# Plot SED
fig, ax = plt.subplots()
ax.loglog(disk.sed_wavs, disk.sed_flux)
ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Flux (Jy)')
ax.set_xlim(5, 2000)
ax.set_ylim(1e-3, 100)
fig.savefig(script_dir / 'SED.png', **savefig_dict)


# Plot astrophysical scene
obs_inclination = 67.52
obs_positionang = 336

# Create image from surface brightness profile
image = pyr.Image.from_disk_surface_brightness_profile(
    S_r=S_23,
    r_au=rbin.mids,
    dist_pc=star.dist_pc,
    scale_height=belt.inc_max_rad,
    inclination=obs_inclination,
    position_angle=obs_positionang,
    resolution=200,
)

# Plot image
ax, _ = image.plot(
    cmap="afmhot",
    vmin=3,
    unit=u.MJy/u.sr,
    axlim_asec=max(image.extent),
)
ax.figure.savefig(script_dir / 'image.png', **savefig_dict)
