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

    @classmethod
    def from_disk_surface_brightness_profile(cls, S_r: u.Quantity, r_au: np.ndarray, 
                                          dist_pc: float, scale_height: float,
                                          inclination: float = 0, position_angle: float = 0,
                                          resolution: int = 200, rave_points_per_pixel=50,
                                          r_mask_au: float = 0) -> 'Image':
        """Create an image from a radial surface brightness profile using RAVE.

        Args:
            S_r (Quantity): Surface brightness profile with units (e.g., Jy/sr)
            r_au (np.ndarray): Radial coordinates (au)
            dist_pc (float): Distance to star (parsec)
            scale_height (float): Disk vertical scale height relative to radius
            inclination (float): Disk inclination (degrees)
            position_angle (float): Position angle E of N (degrees)
            resolution (int): Image size in pixels
            r_mask_au (float): Radius to mask in center (au)

        Returns:
            Image: New image instance
        """
        img = cls(resolution=resolution)
        img.scale_height = scale_height
        img.inclination = inclination
        img.position_angle = position_angle

        # Create bins for RAVE
        r_bounds = np.arange(0, int(resolution/2)+1)
        r_pix = (r_bounds[1:] + r_bounds[:-1]) / 2
        r_model_au = r_pix / r_pix.max() * r_au.max()

        # Interpolate surface brightness onto RAVE bins (strip units for RAVE)
        S_model = np.zeros_like(r_model_au)
        valid_radii = r_model_au >= r_au[0]
        S_model[valid_radii] = np.interp(r_model_au[valid_radii], r_au, S_r.value)

        # Generate RAVE image
        rave_img = rave.MakeImage(
            r_bounds_make=r_bounds,
            weights_make=S_model,
            heights_make=scale_height * r_pix,
            inclination_make=inclination,
            dim=resolution,
            n_points_per_pixel=rave_points_per_pixel,
            kernel=None,
            rapid=False,
            verbose=False
        )

        # Apply central mask if requested
        if r_mask_au > 0:
            r_mask_pix = r_mask_au * resolution/(2 * r_au.max())
            y, x = np.ogrid[-resolution//2:resolution//2, -resolution//2:resolution//2]
            mask = x*x + y*y <= r_mask_pix*r_mask_pix
            rave_img.image[mask] = 0

        # Store data with units preserved
        img.data = rave_img.image * S_r.unit
        extent = r_au.max() / dist_pc * (resolution/2 + 0.5)/(resolution/2)
        img.extent = [-extent, extent, -extent, extent]

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
            unit: u.Unit = u.MJy/u.sr):
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

        # Convert data to requested unit for display
        data_converted = self.data.to(unit)
        
        img_handle = ax.imshow(data_converted.value, extent=self.extent, 
                             origin='lower', cmap=cmap,
                             norm=LogNorm(vmin=vmin, vmax=vmax, clip=True))

        # Add central mask if requested
        if r_mask_asec:
            mask = Circle((0, 0), r_mask_asec, transform=ax.transData, color='black')
            ax.add_patch(mask)

        # Determine colorbar extend based on data range
        if vmax is None:
            extend = 'min'  # Only extend towards vmin if vmax not specified
        else:
            extend = 'both' if np.max(data_converted.value) > vmax else 'min'

        # Configure colorbar
        cbar = plt.colorbar(img_handle, orientation='vertical', ticklocation='left', 
                          shrink=.87, pad=-.1, extend=extend)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=9)
        
        if cticks:
            cbar.ax.set_yticks(cticks)
            cbar.ax.set_yticklabels([str(x) for x in cticks])
        
        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(1)
        cbar.set_label(f'Flux ({unit.to_string()})', color='white', position=(-.4,0), fontsize=10)

        # Configure axes
        ax.set_xticks([])
        ax.set_yticks([])
        if axlim_asec:
            ax.set_xlim(-axlim_asec*.7, axlim_asec*.9)
            ax.set_ylim(-axlim_asec*.8, axlim_asec*.8)

            # Add scale bar
            ax.add_patch(Rectangle((axlim_asec/4, -axlim_asec*.7), 5.0, .1, color='white'))
            ax.text(axlim_asec/4 + 2.5, -axlim_asec*.7 + 0.5, '5\"', color='white', 
                   fontsize=11, verticalalignment='bottom', horizontalalignment='center')
            
            # Add orientation indicators
            ax.add_patch(Rectangle((-axlim_asec*.5, axlim_asec*.6), -2.5, .06, color='grey'))
            ax.add_patch(Rectangle((-axlim_asec*.5, axlim_asec*.6), .06, 2.5, color='grey'))
            ax.text(-axlim_asec*.5 - 2.8, axlim_asec*.6, 'E', color='grey', 
                   fontsize=11, verticalalignment='center', horizontalalignment='right')
            ax.text(-axlim_asec*.5, axlim_asec*.6 + 2.7, 'N', color='grey', 
                   fontsize=11, verticalalignment='bottom', horizontalalignment='center')

        return ax, img_handle
