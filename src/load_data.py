from utils import *

# Install dependencies and mount drive
# ----------------------------------------------------------------------------

from google.colab import drive
drive.mount('/content/drive')

#Import libraries
# ----------------------------------------------------------------------------
import zipfile
import re
import os
import pandas as pd
import numpy as np
from io import BytesIO

print(" Libraries imported successfully")

#Configuration
# ----------------------------------------------------------------------------
# MODIFY THIS PATH to match your Drive location
ZIP_PATH = "/content/drive/MyDrive/CMPE255-NIDSPROJECT/GeneratedLabelledFlows.zip"
OUTPUT_DIR = "/content/drive/MyDrive/CMPE255-NIDSPROJECT/data"

TARGET_FILES = [
    "TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",      # DDoS attacks
    "TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",                  # Multiple DoS variants
    "TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv"                     # Mixed traffic
]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f" Output directory: {OUTPUT_DIR}")
print(f" Will process {len(TARGET_FILES)} files")

# Inspect ZIP contents
# ----------------------------------------------------------------------------
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]

print(f" Found {len(all_csv_files)} CSV files in ZIP\n")
print("Files we'll process:")
for i, f in enumerate(TARGET_FILES, 1):
    print(f"  {i}. {f.split('/')[-1]}")

# Cell 6: Select target CSV file
# ----------------------------------------------------------------------------
# Prioritize files with DoS/DDoS attacks (Wednesday/Friday typically have more attacks)
print("\n" + "="*80)
print("AUTO-DETECTING FILES IN ZIP")
print("="*80)

# First, list ALL files in the ZIP to see the actual structure
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_files_in_zip = zf.namelist()
    csv_files_in_zip = [f for f in all_files_in_zip if f.lower().endswith('.csv')]

    print(f"✓ Found {len(csv_files_in_zip)} CSV files in ZIP:\n")
    for i, f in enumerate(csv_files_in_zip, 1):
        print(f"  {i}. {f}")

# Now match our target files with actual paths in ZIP
print("\n" + "="*80)
print("MATCHING TARGET FILES")
print("="*80)

target_keywords = [
    "Friday-WorkingHours-Afternoon-DDos",
    "Wednesday-workingHours",
    "Tuesday-WorkingHours"
]

matched_files = []
for keyword in target_keywords:
    for csv_file in csv_files_in_zip:
        if keyword.lower() in csv_file.lower():
            matched_files.append(csv_file)
            print(f"✓ Matched '{keyword}' → {csv_file}")
            break
    else:
        print(f"  Could not find file matching '{keyword}'")

print(f"\n✓ Will process {len(matched_files)} files")

# Now process the matched files
print("\n" + "="*80)
print("PROCESSING MULTIPLE CSV FILES")
print("="*80)

all_dataframes = []

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for idx, csv_file in enumerate(matched_files, 1):
        print(f"\n{'='*80}")
        print(f" Processing File {idx}/{len(matched_files)}: {csv_file.split('/')[-1]}")
        print(f"{'='*80}")

        # Load CSV
        try:
            with zf.open(csv_file) as f:
                df_temp = pd.read_csv(f, low_memory=False)

            print(f" Loaded: {df_temp.shape[0]:,} rows × {df_temp.shape[1]} columns")

            # Normalize column names
            orig_cols = df_temp.columns.tolist()
            norm_cols = [normalize_col(c) for c in orig_cols]
            rename_map = dict(zip(orig_cols, norm_cols))
            df_temp.rename(columns=rename_map, inplace=True)

            # Add source file column for tracking
            file_name = csv_file.split('/')[-1].replace('.pcap_ISCX.csv', '')
            df_temp['source_file'] = file_name

            all_dataframes.append(df_temp)
            print(f"✓ Normalized columns and added source tracking")

        except Exception as e:
            print(f" Error loading {csv_file}: {e}")
            continue

if len(all_dataframes) == 0:
    print("\n ERROR: No files were successfully loaded!")
    print("Please check the ZIP file structure.")
else:
    print(f"\n Successfully loaded {len(all_dataframes)} files")

#Load CSV from ZIP
# ----------------------------------------------------------------------------
print(f"\n{'='*80}")
print("COMBINING ALL DATAFRAMES")
print(f"{'='*80}")

# Safety check: ensure we have dataframes to combine
if len(all_dataframes) == 0:
    raise ValueError(" ERROR: No files were successfully loaded! Cannot proceed.")

df = pd.concat(all_dataframes, ignore_index=True)

print(f" Combined {len(all_dataframes)} files")
print(f"  Total shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Show distribution by source file
print("\n Rows per source file:")
print(df['source_file'].value_counts())

