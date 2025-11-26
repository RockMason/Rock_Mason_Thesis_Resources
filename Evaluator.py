
"""
Script to solve RMS of Photometry and Astrometry, Precision
Copyright Rock Mason 2025
"""

import numpy as np
import pandas as pd
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# CHANGE THESE FOR EACH SOFTWARE
software = "AstroImageJ"  # ← change per run
phot_csv = "AIJ_photometry.csv"  # ← your exported photometry table
image_dir = "thesis_test_dataset"
truth = ascii.read("thesis_test_dataset/truth_catalog.csv")

results = []

for i in range(1, 6):
    img = f"science_{i:02d}.fits"
    print(f"Processing {img} with {software}...")

    # Load your exported photometry (has X,Y or RA/Dec + mag)
    phot = ascii.read(phot_csv)  # adjust path/format per software

    # Match by pixel position (tolerance 5 px)
    idx, sep, _ = truth['x', 'y'].match_to_catalog_sky(phot['xcentroid', 'ycentroid'])
    good = sep < 5 * u.pix

    if len(good) < 20:
        print("  Not enough matches!")
        continue

    # Photometry RMS
    delta_mag = phot['mag'][good] - truth['mag'][idx][good]
    rms_mag = np.sqrt(np.mean(delta_mag ** 2))

    # Precision (bright stars, mag < 17)
    bright = truth['mag'][idx][good] < 17
    precision = np.std(delta_mag[bright]) if np.any(bright) else np.nan

    # Astrometry RMS (if WCS exists)
    try:
        w = WCS(f"{image_dir}/calibrated_{i:02d}.fits")  # or your solved file
        ra_dec_solved = w.pixel_to_world(phot['xcentroid'], phot['ycentroid'])
        c1 = SkyCoord(truth['ra'][idx][good], truth['dec'][idx][good], unit='deg')
        c2 = SkyCoord(ra_dec_solved.ra.deg[good], ra_dec_solved.dec.deg[good])
        astrom_rms = np.mean(c1.separation(c2).arcsec)
    except:
        astrom_rms = np.nan

    results.append({
        'Software': software,
        'Image': img,
        'Phot_RMS_mag': round(rms_mag, 4),
        'Phot_Precision_mag': round(precision, 4),
        'Astrom_RMS_arcsec': round(astrom_rms, 3) if not np.isnan(astrom_rms) else "",
        'Blind_Solve_Success': 1,
        'Solve_Time_sec': 120,  # ← type your stopwatch time
        'Total_Time_min': 15.5,  # ← type your total time
        'Calibration_Residual_percent': 0.7,  # measure manually once
        'Notes': ''
    })

# Append to master log
df = pd.DataFrame(results)
df.to_csv("thesis_results_log.csv", mode='a', header=False, index=False)
print("Done! Results appended to thesis_results_log.csv")