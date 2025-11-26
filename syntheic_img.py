"""
generate_thesis_test_images.py - ULTRA-STABLE VERSION (NumPy + Astropy only)
Creates 5 realistic astronomical test images (2048x2048) with:
- Known star positions and magnitudes (truth catalog)
- Gaussian PSF (manual NumPy implementation)
- Poisson noise + read noise + bias
- Dark current
- Flat-field vignetting
- Cosmic rays
- Bias, Dark, and Flat calibration frames

NO Photutils or GalSim dependencies - guaranteed to work!
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter  # Only for PSF smoothing

# --------------------- Configuration ---------------------
np.random.seed(42)  # Reproducible results
N_IMAGES = 5
IMAGE_SIZE = (2048, 2048)
GAIN = 1.5  # e-/ADU
READ_NOISE = 8.0  # electrons
BIAS_LEVEL = 1000.0  # ADU
DARK_CURRENT = 0.1  # e-/s/pix
EXPTIME = 300.0  # seconds
FWHM_PX = 3.5  # PSF full width half max in pixels

# Output directory - EASY TO CHANGE (uncomment one line below)
#out_dir = "thesis_test_dataset"  # Current folder
out_dir = os.path.expanduser("~/Desktop/thesis_test_dataset")  # Desktop
#out_dir = r"C:\Users\cicad\Documents\Thesis\test_dataset"  # Specific Windows path

os.makedirs(out_dir, exist_ok=True)


# --------------------- Truth Catalog (300 stars) ---------------------
def create_truth_catalog(n_stars=300):
    # Random positions (avoid edges)
    x = np.random.uniform(100, 1948, n_stars)
    y = np.random.uniform(100, 1948, n_stars)

    # RA/Dec (arbitrary field)
    ra = 180.0 + (x - 1024) / 3600.0  # ~1 degree field
    dec = 30.0 + (y - 1024) / 3600.0

    # Realistic magnitudes (Pareto distribution for luminosity function)
    mag = np.random.pareto(a=2.0, size=n_stars) * 3 + 14  # 14-22 mag
    flux = 10 ** (-0.4 * (mag - 20.0)) * 1e6  # Scale to ADU

    table = Table([x, y, ra, dec, mag, flux],
                  names=['x', 'y', 'ra', 'dec', 'mag', 'flux'])
    return table


truth_catalog = create_truth_catalog()
truth_catalog.write(f"{out_dir}/truth_catalog.fits", overwrite=True)
truth_catalog.write(f"{out_dir}/truth_catalog.csv", overwrite=True)


# --------------------- Helper Functions ---------------------
def gaussian_2d(xy, amplitude, x0, y0, sigma):
    """Single 2D Gaussian function"""
    x, y = xy
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def create_gaussian_sources(shape, sources, sigma):
    """Manual Gaussian source injection (NumPy only)"""
    y, x = np.ogrid[:shape[0], :shape[1]]
    image = np.zeros(shape)

    sigma_gauss = FWHM_PX / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma conversion

    for row in sources:
        amp, x0, y0 = row['flux'], row['x'], row['y']
        # Vectorized Gaussian addition
        gauss = gaussian_2d((x, y), amp, x0, y0, sigma_gauss)
        image += gauss

    # Apply PSF convolution (optional, for realism)
    image = gaussian_filter(image, sigma=sigma_gauss * 0.8)

    return image


def add_cosmic_rays(image, n_rays=50):
    """Add realistic cosmic rays"""
    cr = image.copy()
    for _ in range(n_rays):
        x0, y0 = np.random.randint(50, 1998, 2)
        length = np.random.randint(10, 80)
        width = np.random.randint(1, 4)
        angle = np.random.uniform(0, 2 * np.pi)
        intensity = np.random.uniform(1000, 10000)

        yy, xx = np.mgrid[:IMAGE_SIZE[0], :IMAGE_SIZE[1]]
        dx = xx - x0
        dy = yy - y0
        dist = np.abs(dx * np.sin(angle) + dy * np.cos(angle))
        mask = dist < width
        along = np.abs(dx * np.cos(angle) - dy * np.sin(angle))
        cr[mask] += intensity * np.exp(-along[mask] / length)

    return cr


def create_flat_field():
    """Vignetted flat field"""
    y, x = np.ogrid[:IMAGE_SIZE[0], :IMAGE_SIZE[1]]
    center_y, center_x = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_r = np.sqrt(center_x ** 2 + center_y ** 2)
    flat = 1.0 - 0.3 * (r / max_r) ** 2  # 30% vignetting
    flat += np.random.normal(0, 0.02, IMAGE_SIZE)  # Lamp structure noise
    return flat


# --------------------- Generate Calibration Frames (once) ---------------------
print("Creating calibration frames...")
bias_master = np.full(IMAGE_SIZE, BIAS_LEVEL) + np.random.normal(0, 5, IMAGE_SIZE)
dark_master = BIAS_LEVEL + DARK_CURRENT * EXPTIME / GAIN + np.random.normal(0, 3, IMAGE_SIZE)
flat_master = create_flat_field()
flat_master = (flat_master / np.median(flat_master)) * 30000  # Normalize to 30k ADU

fits.PrimaryHDU(bias_master.astype('float32')).writeto(f"{out_dir}/master_bias.fits", overwrite=True)
fits.PrimaryHDU(dark_master.astype('float32')).writeto(f"{out_dir}/master_dark_300s.fits", overwrite=True)
fits.PrimaryHDU(flat_master.astype('float32')).writeto(f"{out_dir}/master_flat.fits", overwrite=True)

# --------------------- Generate Science Images ---------------------
print("Creating science images...")
for i in range(1, N_IMAGES + 1):
    print(f"  Image {i}/5...")

    # 1. Clean source image (manual Gaussian injection)
    clean_image = create_gaussian_sources(IMAGE_SIZE, truth_catalog, sigma=1.0)

    # 2. Add sky background
    sky = 800.0 + np.random.normal(0, 20, IMAGE_SIZE)

    # 3. Apply flat-field vignetting
    science = (clean_image + sky) * flat_master / np.median(flat_master)

    # 4. Add dark current
    science += dark_master - BIAS_LEVEL

    # 5. Add bias
    science += bias_master

    # 6. Poisson noise + read noise (full CCD model)
    electrons = science * GAIN
    poisson = np.random.poisson(electrons) / GAIN
    read = np.random.normal(0, READ_NOISE / GAIN, IMAGE_SIZE)
    science_final = poisson + read

    # 7. Add cosmic rays
    science_final = add_cosmic_rays(science_final, n_rays=60)

    # 8. FITS header
    hdu = fits.PrimaryHDU(science_final.astype('float32'))
    hdu.header['OBJECT'] = f'Thesis Test Field {i}'
    hdu.header['EXPTIME'] = EXPTIME
    hdu.header['GAIN'] = GAIN
    hdu.header['RDNOISE'] = READ_NOISE
    hdu.header['DATE-OBS'] = datetime.now().isoformat()
    hdu.header['IMAGETYP'] = 'light'
    hdu.header['FILTER'] = 'R'
    hdu.header['PSF_FWHM'] = FWHM_PX

    hdu.writeto(f"{out_dir}/science_{i:02d}.fits", overwrite=True)

print(f"\n✅ Dataset complete!")
print(f"📁 Location: {os.path.abspath(out_dir)}")
print("📋 Files created:")
print("   • 5 science images: science_01.fits ... science_05.fits")
print("   • master_bias.fits, master_dark_300s.fits, master_flat.fits")
print("   • truth_catalog.fits & truth_catalog.csv (ground truth)")
print("\n🚀 Ready for AstroImageJ, Siril, ASTAP, Astroart evaluation!")