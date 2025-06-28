#!/usr/bin/env python
# Script to oversample rare classes in the CIC-IDS-2017 dataset

import pandas as pd
import numpy as np
from collections import Counter
from datetime import timedelta
import random
from tqdm import tqdm
import os
import sys

def get_dataset_path():
    """Find the dataset file."""
    possible_paths = [
        "CIC-IDS-2017(unified).csv",
        "/kaggle/input/dataset/CIC-IDS-2017(unified).csv",
        "/kaggle/working/CIC-IDS-2017(unified).csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Unified dataset file not found in any of the expected locations.")

def sequence_smote(df, rare_class, sequence_length=5, n_samples=1000):
    """
    Generate synthetic sequences for rare classes using a simple SMOTE-like approach.
    
    Args:
        df: DataFrame with original data
        rare_class: Label of the rare class to oversample
        sequence_length: Length of sequences to generate
        n_samples: Number of synthetic samples to generate
        
    Returns:
        DataFrame with synthetic samples
    """
    # Get all samples of the rare class
    rare_samples = df[df['Label'] == rare_class]
    
    if len(rare_samples) < 2:
        print(f"Warning: Not enough samples for class {rare_class} to perform oversampling")
        return pd.DataFrame()
    
    # Find sequences by looking at consecutive rows with timestamps
    df_with_dt = rare_samples.copy()
    df_with_dt['Timestamp'] = pd.to_datetime(df_with_dt['Timestamp'], errors='coerce')
    df_with_dt = df_with_dt.sort_values('Timestamp')
    
    # Get all features except Label and Timestamp
    feature_cols = [col for col in df.columns if col not in ['Label', 'Timestamp']]
    
    # Identify numeric and non-numeric columns
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        try:
            # Check if column can be converted to numeric
            pd.to_numeric(df_with_dt[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)
    
    print(f"Processing {rare_class}: {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns")
    
    synthetic_samples = []
    
    for _ in tqdm(range(n_samples), desc=f"Generating samples for {rare_class}"):
        # Randomly select a sequence of records
        if len(df_with_dt) <= sequence_length:
            # If we don't have enough records, use random sampling with replacement
            sequence = df_with_dt.sample(sequence_length, replace=True)
        else:
            # Try to find an actual sequence of consecutive records
            start_idx = random.randint(0, len(df_with_dt) - sequence_length)
            sequence = df_with_dt.iloc[start_idx:start_idx + sequence_length]
        
        # Sort by timestamp
        sequence = sequence.sort_values('Timestamp')
        
        # Generate a new synthetic sample by averaging feature values
        synthetic_record = {}
        
        # Process numeric columns - average with small noise
        for col in numeric_cols:
            try:
                col_values = pd.to_numeric(sequence[col], errors='coerce')
                col_values = col_values.dropna()
                
                if len(col_values) > 0:
                    col_mean = col_values.mean()
                    col_std = col_values.std() if len(col_values) > 1 else 0.001
                    noise = np.random.normal(0, 0.01 * col_std if col_std > 0 else 0.001)
                    synthetic_record[col] = col_mean + noise
                else:
                    # Use the most common value in the original dataset for this column
                    synthetic_record[col] = df_with_dt[col].mode()[0] if not df_with_dt[col].mode().empty else 0
            except:
                # Fallback for any issues
                synthetic_record[col] = df_with_dt[col].iloc[0] if not df_with_dt.empty else 0
        
        # Process categorical columns - use most frequent value or random choice
        for col in categorical_cols:
            try:
                # For categorical columns, use the most frequent value from the sequence
                value_counts = sequence[col].value_counts()
                if not value_counts.empty:
                    synthetic_record[col] = value_counts.index[0]
                else:
                    # Fallback to a random choice from the original class data
                    synthetic_record[col] = random.choice(df_with_dt[col].dropna().tolist() if not df_with_dt[col].dropna().empty else [''])
            except:
                # Fallback for any issues
                synthetic_record[col] = df_with_dt[col].iloc[0] if not df_with_dt.empty else ''
        
        # Set the label
        synthetic_record['Label'] = rare_class
        
        # Set timestamp with small jitter to preserve time pattern
        try:
            base_timestamp = sequence['Timestamp'].iloc[-1]
            if pd.notna(base_timestamp):
                time_increment = random.randint(1, 1000)  # milliseconds
                synthetic_record['Timestamp'] = (base_timestamp + timedelta(milliseconds=time_increment)).strftime('%m/%d/%Y %I:%M:%S %p')
            else:
                # Use a random timestamp from the original data
                synthetic_record['Timestamp'] = df_with_dt['Timestamp'].sample(1).iloc[0].strftime('%m/%d/%Y %I:%M:%S %p')
        except:
            # If timestamp handling fails, use the first timestamp from the original data
            try:
                synthetic_record['Timestamp'] = df_with_dt['Timestamp'].iloc[0].strftime('%m/%d/%Y %I:%M:%S %p')
            except:
                synthetic_record['Timestamp'] = '01/01/2017 12:00:00 AM'  # Default fallback
        
        synthetic_samples.append(synthetic_record)
    
    return pd.DataFrame(synthetic_samples)

def main():
    print("Starting dataset balancing process...")
    
    # Find the dataset
    dataset_path = get_dataset_path()
    print(f"Using dataset at: {dataset_path}")
    
    # Load the dataset (in chunks to manage memory)
    print("Loading dataset...")
    df_chunks = []
    for chunk in tqdm(pd.read_csv(dataset_path, chunksize=100000)):
        df_chunks.append(chunk)
    
    df = pd.concat(df_chunks)
    print(f"Dataset loaded: {len(df)} records")
    
    # Try to ensure timestamp is properly formatted
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        # Convert back to string format to ensure consistency
        df['Timestamp'] = df['Timestamp'].dt.strftime('%m/%d/%Y %I:%M:%S %p')
    except Exception as e:
        print(f"Warning: Error processing timestamps: {e}")
        # We'll keep going anyway, as this doesn't affect the class balancing
    
    # Get class distribution
    class_counts = Counter(df['Label'])
    total_records = len(df)
    
    print("\nOriginal class distribution:")
    for cls, count in class_counts.items():
        percentage = (count / total_records) * 100
        print(f"{cls}: {count} samples ({percentage:.4f}%)")
    
    # Determine oversampling strategy for each class
    oversampling_targets = {}
    synthetic_samples = []
    
    print("\nDetermining oversampling strategy for each class...")
    for cls, count in class_counts.items():
        percentage = (count / total_records) * 100
        
        # Very rare classes (< 0.1%)
        if percentage < 0.1:
            # Limit to max 5000 samples for very rare classes to avoid overwhelming the system
            target_percentage = min(1.0, percentage * 10)  # Try to increase by 10x but cap at 1%
            target_count = min(5000, int(total_records * target_percentage / 100))
            oversampling_targets[cls] = max(0, target_count - count)
            print(f"{cls}: Very rare class ({percentage:.4f}%), oversampling to {target_percentage:.2f}% ({oversampling_targets[cls]} new samples)")
        
        # Moderate classes (0.1% - 5%)
        elif percentage < 5.0:
            # Limit to max 50000 samples for moderate classes
            target_percentage = min(5.0, percentage * 2)  # At least doubling but capped at 5%
            target_count = min(50000, int(total_records * target_percentage / 100))
            oversampling_targets[cls] = max(0, target_count - count)
            print(f"{cls}: Moderate class ({percentage:.4f}%), oversampling to {target_percentage:.1f}% ({oversampling_targets[cls]} new samples)")
        
        # Large classes (> 5%)
        else:
            oversampling_targets[cls] = 0
            print(f"{cls}: Large class ({percentage:.4f}%), no oversampling needed")
    
    # Generate synthetic samples for classes that need oversampling
    print("\nGenerating synthetic samples...")
    for cls, samples_to_generate in oversampling_targets.items():
        if samples_to_generate > 0:
            # Cap the number of samples to generate at a reasonable limit
            # to avoid memory issues or excessive processing time
            samples_to_generate = min(samples_to_generate, 10000)
            print(f"Will generate {samples_to_generate} samples for class {cls} (capped for performance)")
            
            synthetic_df = sequence_smote(
                df, 
                cls, 
                sequence_length=5, 
                n_samples=samples_to_generate
            )
            
            if not synthetic_df.empty:
                synthetic_samples.append(synthetic_df)
                print(f"Successfully generated {len(synthetic_df)} synthetic samples for class {cls}")
    
    # Combine original and synthetic data
    if synthetic_samples:
        synthetic_df_all = pd.concat(synthetic_samples, ignore_index=True)
        print(f"Total synthetic samples generated: {len(synthetic_df_all)}")
        
        final_df = pd.concat([df, synthetic_df_all], ignore_index=True)
    else:
        final_df = df
        print("No synthetic samples were generated")
    
    # Sort by timestamp to maintain chronological order
    try:
        print("Sorting by timestamp...")
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], errors='coerce')
        # Remove rows with invalid timestamps if needed
        final_df = final_df.dropna(subset=['Timestamp'])
        final_df = final_df.sort_values('Timestamp')
        final_df['Timestamp'] = final_df['Timestamp'].dt.strftime('%m/%d/%Y %I:%M:%S %p')
    except Exception as e:
        print(f"Warning: Could not sort by timestamp: {e}")
        print("Proceeding without timestamp sorting.")
    
    # Save the balanced dataset
    output_path = "cic_ids2017_oversampled.csv"
    print(f"Saving balanced dataset to {output_path}...")
    final_df.to_csv(output_path, index=False)
    
    # Print final statistics
    final_class_counts = Counter(final_df['Label'])
    final_total = len(final_df)
    
    print("\nFinal class distribution:")
    for cls, count in final_class_counts.items():
        original_count = class_counts.get(cls, 0)
        percentage = (count / final_total) * 100
        increase = ((count / original_count) - 1) * 100 if original_count > 0 else float('inf')
        print(f"{cls}: {count} samples ({percentage:.4f}%) - {increase:.1f}% increase")
    
    print(f"\nOriginal dataset: {total_records} records")
    print(f"Final balanced dataset: {final_total} records")
    print(f"Added {final_total - total_records} synthetic records")
    print("Balancing complete!")

if __name__ == "__main__":
    main() 