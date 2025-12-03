import pandas as pd
import numpy as np
import os

# CHANGE THIS TO YOUR FOLDER (use forward slashes or raw string)
folder = r"C:\Users\cicad\Desktop\thesis_test_dataset\AIJ_calibrated"  # your folder here

os.chdir(folder)  # sets working directory
print(f"Working directory set to: {os.getcwd()}\n")

# Find all AIJ photometry CSVs
csv_files = [f for f in os.listdir() if f.startswith("AIJ_phot_") and f.endswith(".csv")]
csv_files.sort()  # ensures 01, 02, 03, 04, 05 order

if len(csv_files) == 0:
    print("No AIJ_phot_*.csv files found — check folder and filenames")
    exit()

print(f"Found {len(csv_files)} files: {csv_files}\n")

rms_values = []

for file in csv_files:
    df = pd.read_csv(file)

    # Find comparison star columns
    rel_cols = [col for col in df.columns if 'rel_flux_C' in col]
    fluxes = df[rel_cols].iloc[0].values

    rms_rel = np.std(fluxes)
    rms_mag = -2.5 * np.log10(1 + rms_rel)

    rms_values.append(rms_mag)

    print(f"{file:25} → RMS = {rms_mag:.7f} mag  ({rms_mag * 1000:.3f} mmag)")

# Final result
avg_rms = np.mean(rms_values)
std_rms = np.std(rms_values)

print("\n" + "=" * 60)
print(f"FINAL ASTROIMAGEJ RESULT (n = {len(rms_values)} images)")
print(f"Photometric RMS = {avg_rms:.7f} ± {std_rms:.7f} mag")
print(f"                = {avg_rms * 1000:.3f} ± {std_rms * 1000:.3f} mmag")
print("=" * 60)

# Save to file for thesis
with open("AIJ_FINAL_RMS.txt", "w") as f:
    f.write(f"AstroImageJ Photometric RMS: {avg_rms:.7f} mag ({avg_rms * 1000:.3f} mmag)\n")
print("\nResult saved to AIJ_FINAL_RMS.txt")