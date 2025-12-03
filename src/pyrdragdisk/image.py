import numpy as np
import scipy.ndimage
import rave
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
import astropy.units as u
from astropy.nddata import block_reduce

class Image:
    """Class representing a 2D image of a debris disk.
    
    Attributes:
        data (Quantity): Image data with units (e.g., Jy/sr)
        resolution (int): Image resolution in pixels
        extent (tuple): Image extent (xmin, xmax, ymin, ymax) in arcsec
        position_angle (float): Position angle in degrees E of N
        inclination (float): Inclination angle in degrees
        scale_height (float): Vertical scale height relative to radius
    """
    def __init__(self, resolution: int = 200):
        self.data = None
        self.resolution = resolution
        self.extent = None
        self.position_angle = 0
        self.inclination = 0
        self.scale_height = 0

    @property
    def pixel_scale(self) -> u.Quantity:
        """Get the pixel scale in arcseconds per pixel."""
        if self.extent is None:
            raise ValueError("Image extent not set")
        return (2 * max(self.extent) / self.data.shape[0]) * u.arcsec

    @classmethod
    def from_disk_surface_brightness_profile(cls, S_r: u.Quantity, r_au: np.ndarray, 
                                          dist_pc: float, scale_height: float,
                                          inclination: float = 0, position_angle: float = 0,
                                          resolution: int = 200, rave_points_per_pixel=10,
                                          pixel_scale: u.Quantity | None = None,
                                          fov: u.Quantity | None = None) -> 'Image':
        """Create an image from a radial surface brightness profile using RAVE.

        Args:
            S_r (Quantity): Surface brightness profile with units (e.g., Jy/sr)
            r_au (np.ndarray): Radial coordinates (au)
            dist_pc (float): Distance to star (parsec)
            scale_height (float): Disk vertical scale height relative to radius
            inclination (float): Disk inclination (degrees)
            position_angle (float): Position angle E of N (degrees)
            resolution (int): Image size in pixels (ignored if fov is provided)
            pixel_scale (Quantity, optional): Pixel scale with unit either arcsec or au (per pixel).
            fov (Quantity, optional): Desired total field of view (width = height) in arcsec or au.
            r_mask_au (float): Radius to mask in center (au)

        Returns:
            Image: New image instance
        """
        # --- Early validation and conversion to au ---
        pxs_au = None    # au per pixel (float)
        fov_au = None   # total FOV in au (float)

        if pixel_scale is not None:
            if not isinstance(pixel_scale, u.Quantity):
                raise TypeError("pixel_scale must be an astropy Quantity")
            if not (pixel_scale.unit.is_equivalent(u.arcsec) or pixel_scale.unit.is_equivalent(u.au)):
                raise ValueError("pixel_scale must have units of arcsec or au")
            if pixel_scale.unit.is_equivalent(u.au):
                pxs_au = pixel_scale.to_value(u.au)  # au/pixel
            else:
                # arcsec/pixel -> au/pixel using small-angle relation (1 arcsec at 1 pc = 1 au)
                pxs_au = pixel_scale.to_value(u.arcsec) * dist_pc  # au/pixel
        
        if fov is not None:
            if not isinstance(fov, u.Quantity):
                raise TypeError("fov must be an astropy Quantity")
            if  not (fov.unit.is_equivalent(u.arcsec) or fov.unit.is_equivalent(u.au)):
                raise ValueError("fov must have units of arcsec or au")            
            if pixel_scale is None:
                raise TypeError("pixel_scale is required when fov is provided")
            if fov.unit.is_equivalent(u.au):
                fov_au = fov.to_value(u.au)
            else:
                fov_au = fov.to_value(u.arcsec) * dist_pc  # au
            
            dim = int(np.round(fov_au / pxs_au))
            if dim < 1:
                raise ValueError("Computed image dimension from fov/pixel_scale must be >= 1")
            elif np.mod(dim, 2) == 1:
                dim += 1  # make dimension even
        else:
            if resolution is None:
                raise ValueError("Either fov or resolution must be provided")
            dim = resolution
            if pixel_scale is None:
                pxs_au = r_au.max() / (dim / 2)
            fov_au = dim * pxs_au

        # Instantiate image with the effective resolution
        img = cls(resolution=dim)
        img.scale_height = scale_height
        img.inclination = inclination
        img.position_angle = position_angle

        # Generate RAVE binning
        has_model_flux = S_r.value > 0
        within_fov = r_au / np.sin(np.deg2rad(inclination)) < fov_au / 2
        n_rbin_rave = int(np.floor(r_au[has_model_flux & within_fov].max() / pxs_au))
        n_rbin_rave = int(np.floor(r_au.max() / pxs_au))
        if n_rbin_rave < 1:
            raise ValueError(f"Number of RAVE bins must be > 1 (current: {n_rbin_rave}); "
                             f"decrease pixel_scale.")
        elif n_rbin_rave > 2000:
            raise ValueError(f"Number of RAVE bins must be < 2000 (current: {n_rbin_rave}); "
                             f"increase pixel_scale.")
        elif n_rbin_rave > 500:
            print(f"Warning: Number of RAVE bins is quite large ({n_rbin_rave}); "
                  f"this may lead to long computation times. Consider increasing pixel_scale.")
        r_pix_bounds = np.arange(0, n_rbin_rave+1)
        r_pix_mids = (r_pix_bounds[1:] + r_pix_bounds[:-1]) / 2
        r_pix_mids_au = r_pix_mids * pxs_au
        extent_arcsec = (dim / 2 + .5) * (pxs_au / dist_pc)

        # Interpolate surface brightness onto RAVE bins (strip units for RAVE)
        S_pix = np.zeros_like(r_pix_mids_au)
        valid_radii = r_pix_mids_au >= r_au[0]
        S_pix[valid_radii] = np.interp(r_pix_mids_au[valid_radii], r_au, S_r.value)

        # Generate RAVE image
        rave_img = rave.MakeImage(
            r_bounds_make=r_pix_bounds,
            weights_make=S_pix,
            heights_make=scale_height * r_pix_mids,
            inclination_make=inclination,
            dim=dim,
            n_points_per_pixel=rave_points_per_pixel,
            kernel=None,
            rapid=False,
            verbose=False
        )

        # Store data with units preserved
        img.data = rave_img.image * S_r.unit
        img.extent = [-extent_arcsec, extent_arcsec, -extent_arcsec, extent_arcsec]

        # Apply position angle rotation if needed (preserve units)
        if position_angle != 90:
            img.data = scipy.ndimage.rotate(img.data.value, -position_angle+90, 
                                          reshape=False, mode='constant', order=1) * S_r.unit

        return img

    def downsample(self, factor: int) -> 'Image':
        """Downsample the image by a given factor using block reduction.
        
        Args:
            factor (int): Downsampling factor. Original dimensions must be divisible by this factor.
            
        Returns:
            Image: New downsampled image instance
        
        Note:
            Uses mean combination of pixels within each block.
            Field of view (extent) remains unchanged while resolution decreases.
        """
        if not isinstance(factor, int) or factor < 1:
            raise ValueError("Downsampling factor must be a positive integer")
            
        if self.data is None:
            raise ValueError("No data to downsample")
            
        if self.data.shape[0] % factor != 0 or self.data.shape[1] % factor != 0:
            raise ValueError(f"Image dimensions ({self.data.shape}) must be divisible by factor {factor}")
        
        # Create new image instance
        downsampled = Image(resolution=self.resolution // factor)
        
        # Downsample the data, preserving units
        downsampled.data = block_reduce(self.data, block_size=(factor, factor), func=np.mean)
        
        # Copy attributes (extent remains the same as it represents physical size)
        downsampled.extent = self.extent
        downsampled.position_angle = self.position_angle
        downsampled.inclination = self.inclination
        downsampled.scale_height = self.scale_height
        
        return downsampled

    def copy(self) -> 'Image':
        """Create a deep copy of the Image instance.
        
        Returns:
            Image: New copy of the image with all attributes duplicated
        """
        return copy.deepcopy(self)

    def plot(self, ax=None, cmap='afmhot', vmin=None, vmax=None, 
            axlim_asec=None, r_mask_asec=None, cticks=None,
            contour_levels=[], unit: u.Unit = u.MJy/u.sr,
            title=None, cbar_label=None, norm=None,
            compass_size=2.5, show_cbar=True,
            beam_rad=None) -> tuple:
        """Plot the image with scientific annotations.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            cmap (str): Colormap name
            vmin (float, optional): Minimum value for color scaling
            vmax (float, optional): Maximum value for color scaling
            axlim_asec (float, optional): Axis limits in arcsec
            r_mask_asec (float, optional): Radius of central mask in arcsec
            cticks (array-like, optional): Custom colorbar ticks
            unit (Unit): Output unit for display (default: MJy/sr)
        
        Returns:
            tuple: (ax, image_handle)
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.set_facecolor('black')

        # Convert data to requested unit for display
        data_converted = self.data.to(unit)

        # Retrieve colormap and set NaN color to nan-color
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(color='magenta', alpha=0.5)
        
        if norm is None:
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            
        img_handle = ax.imshow(data_converted.value, extent=self.extent, 
                             origin='lower', cmap=cmap,
                             norm=norm)

        # Add central mask if requested
        if r_mask_asec:
            mask = Circle((0, 0), r_mask_asec, transform=ax.transData, color='black')
            ax.add_patch(mask)

        # Determine colorbar extend based on data range
        # if vmax is None:
        #     extend = 'min'  # Only extend towards vmin if vmax not specified
        # else:
        #     extend = 'both' if np.max(data_converted.value) > vmax else 'min'
        if np.nanmin(data_converted.value) < norm.vmin and np.nanmax(data_converted.value) > norm.vmax:
            extend = 'both'
        elif np.nanmin(data_converted.value) < norm.vmin:
            extend = 'min'
        elif np.nanmax(data_converted.value) > norm.vmax:
            extend = 'max'
        else:
            extend = 'neither'

        # Add contour lines
        if len(contour_levels) > 0:
            contour_lines = ax.contour(data_converted.value, levels=contour_levels, 
                                    origin='lower', extent=self.extent,
                                    colors='black', alpha=0.5, linewidths=.5)
            
        # Configure colorbar
        if show_cbar:
            cbar = plt.colorbar(img_handle, orientation='vertical', 
                            shrink=.87, pad=-.08, extend=extend)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_tick_params(which='major', colors='white')
            cbar.ax.yaxis.set_tick_params(which='minor', colors='white', labelleft=False)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=9)
            if cticks:
                cbar.ax.set_yticks(cticks)
                cbar.ax.set_yticklabels([str(x) for x in cticks])
            
            cbar.outline.set_edgecolor('white')
            cbar.outline.set_linewidth(1)
            if cbar_label == None:
                cbar.set_label(f'Flux ({unit.to_string()})', color='white', position=(-.4,0), fontsize=10)
            else:
                cbar.set_label(cbar_label, color='white', position=(-.4,0), fontsize=10)
            
            if len(contour_levels) > 0:
                cbar.add_lines(contour_lines)
        
        # Configure axes
        ax.set_xticks([])
        ax.set_yticks([])
        if axlim_asec is None:
            axlim_asec = max(self.extent)

        ax.set_xlim(-axlim_asec*.7, axlim_asec*.9)
        ax.set_ylim(-axlim_asec*.8, axlim_asec*.8)

        # Add scale bar
        ax.add_patch(Rectangle((axlim_asec/5, -axlim_asec*.7), 5.0, .1, color='white'))
        ax.text(axlim_asec/5 + 2.5, -axlim_asec*.7 + 0.5, '5\"', color='white', 
                fontsize=11, verticalalignment='bottom', horizontalalignment='center')
        
        # Add orientation indicators
        compass_size = 2.5 * axlim_asec / 20
        ax.add_patch(Rectangle((-axlim_asec*.45, axlim_asec*.5), -compass_size, .06, color='grey'))
        ax.add_patch(Rectangle((-axlim_asec*.45, axlim_asec*.5), .06, compass_size, color='grey'))
        ax.text(-axlim_asec*.45 - compass_size*1.12, axlim_asec*.5, 'E', color='grey', 
            fontsize=11, verticalalignment='center', horizontalalignment='right')
        ax.text(-axlim_asec*.45, axlim_asec*.5 + compass_size*1.08, 'N', color='grey', 
            fontsize=11, verticalalignment='bottom', horizontalalignment='center')
        
        # Add beam size indicator if provided
        if beam_rad is not None:
            ax.add_patch(Circle((-axlim_asec*.55, -axlim_asec*.67), beam_rad, 
                                edgecolor='white', facecolor='none', lw=1))
            ax.text(-axlim_asec*.55 + beam_rad*2.5, -axlim_asec*.681, 'beam', 
                    color='silver', fontsize=11, ha='left', va='center')

        # Add title 
        if title is not None:
            # ax.text(0.5, 1.0, title,
            #     color='white', ha='center', va='top', transform=ax.transAxes, fontsize=10
            # ) # old
            ax.text(0.5, 0.95, title,
                color='white', ha='center', va='top', transform=ax.transAxes, fontsize=14
            )
        
        return ax, img_handle
    
    def add_star(self, wav: float, star=None, stellar_flux_density=None) -> 'Image':
        """Add stellar flux to the central pixel of the image.
        
        Args:
            wav (float): Wavelength in microns at which to calculate stellar flux
            star: Star instance (pyrdragdisk.Star) containing stellar properties and distance
            stellar_flux_density (Quantity, optional): Directly provide stellar spectral flux density (W/m²/μm)
            
        Returns:
            Image: New image instance with stellar flux added to central pixel
            
        Note:
            The flux is converted to surface brightness units consistent with the image data.
        """
        if self.data is None:
            raise ValueError("No image data to add star to")
            
        if star is not None and stellar_flux_density is None:
            if not hasattr(star, '_optprops_star') or star._optprops_star is None:
                raise ValueError("Star must have an optprops_star to get spectral flux density")
            # Get stellar spectral flux density at the specified wavelength
            stellar_flux_density = star.get_spectral_flux_density(wav, to_jy=True, distance=star.dist_pc * u.pc)
        elif star is None and stellar_flux_density is None:
            raise ValueError("Either star or stellar_flux_density must be provided")

        
        # Convert flux density (W/m²/μm) to surface brightness at pixel scale
        # Stellar flux is received as a point source, so we need to convert to surface brightness
        pixel_area_sr = self.pixel_scale.to(u.rad)**2  # pixel area in steradians
        
        # Convert to surface brightness by dividing by pixel solid angle
        stellar_surface_brightness = (stellar_flux_density / pixel_area_sr).to(self.data.unit)
        
        # Add to central pixel, or pixels if the image has even dimensions
        ny, nx = self.data.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0  # exact optical center in pixel coords (centers at integers)

        # Integer neighbors
        y0, x0 = int(np.floor(cy)), int(np.floor(cx))
        y1, x1 = min(y0 + 1, ny - 1), min(x0 + 1, nx - 1)

        # Decide by parity of dimensions
        odd_y = (ny % 2 == 1)
        odd_x = (nx % 2 == 1)

        if odd_y and odd_x:
            # Single central pixel
            self.data[y0, x0] += stellar_surface_brightness
        elif (not odd_y) and (not odd_x):
            # Four central pixels, equal quarters
            w = 0.25
            self.data[y0, x0] += stellar_surface_brightness * w
            self.data[y0, x1] += stellar_surface_brightness * w
            self.data[y1, x0] += stellar_surface_brightness * w
            self.data[y1, x1] += stellar_surface_brightness * w
        elif odd_y and (not odd_x):
            # Two central columns (vertical boundary), split 50/50 on the central row
            w = 0.5
            self.data[y0, x0] += stellar_surface_brightness * w
            self.data[y0, x1] += stellar_surface_brightness * w
        else:
            # (not odd_y) and odd_x: two central rows (horizontal boundary), split 50/50 on the central column
            w = 0.5
            self.data[y0, x0] += stellar_surface_brightness * w
            self.data[y1, x0] += stellar_surface_brightness * w