#!/usr/bin/env python
# Python script to unify the dataset classes by converting "Attempted" variants to their parent classes

import csv
import sys
from tqdm import tqdm

# Increase CSV field size limit to handle large files
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

print("Starting to process the dataset...")
print("This may take some time due to the large file size...")

# Map of classes to unify
class_map = {
    "DoS Slowhttptest - Attempted": "DoS Slowhttptest",
    "DoS slowloris - Attempted": "DoS slowloris",
    "Bot - Attempted": "Bot",
    "Web Attack - Brute Force - Attempted": "Web Attack - Brute Force",
    "Web Attack - XSS - Attempted": "Web Attack - XSS",
    "DoS Hulk - Attempted": "DoS Hulk",
    "DoS GoldenEye - Attempted": "DoS GoldenEye",
    "Infiltration - Attempted": "Infiltration",
    "FTP-Patator - Attempted": "FTP-Patator",
    "SSH-Patator - Attempted": "SSH-Patator"
}

# Total records from class_distribution.txt
total_records = 2100814
processed_records = 0
skipped_records = 0

# Process the file
try:
    with open('CIC-IDS-2017.csv', 'r', newline='', encoding='utf-8', errors='replace') as infile, \
         open('CIC-IDS-2017(unified).csv', 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header
        header = next(reader)
        writer.writerow(header)
        
        # Process each row with a progress bar
        for row in tqdm(reader, total=total_records-1, desc="Processing records"):
            try:
                # Skip empty rows
                if not row:
                    skipped_records += 1
                    continue
                
                # Get the label (last column)
                label = row[-1]
                
                # Check if the label needs to be unified
                if label in class_map:
                    row[-1] = class_map[label]
                
                # Write the processed row
                writer.writerow(row)
                processed_records += 1
                
            except Exception as e:
                print(f"Error processing row: {e}")
                skipped_records += 1
                continue
except Exception as e:
    print(f"Error: {e}")

print(f"Processing complete. Unified dataset saved as 'CIC-IDS-2017(unified).csv'")
print(f"Processed records: {processed_records}")
print(f"Skipped records: {skipped_records}")
