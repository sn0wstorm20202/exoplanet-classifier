#!/usr/bin/env python3
"""
NASA Exoplanet Data Processing Script

This script loads and combines three NASA datasets:
- k2_data.csv (K2 mission data)
- kepler_data.csv (Kepler mission data)
- tess_data.csv (TESS mission data)

It performs the following operations:
1. Inspects column names and data types
2. Maps different column name formats to standardized features
3. Extracts and standardizes labels
4. Handles missing values and outliers
5. Saves processed data to processed_combined.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def inspect_csv_structure(filepath):
    """Inspect CSV file structure and return column information."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print('='*60)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        # Try reading with different encodings and skip comment lines
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, comment='#', nrows=5)
                print(f"✅ Successfully read with encoding: {encoding}")
                break
            except Exception as e:
                continue
        
        if df is None:
            print("❌ Could not read file with any encoding")
            return None
        
        # Full dataset for analysis
        df_full = pd.read_csv(filepath, encoding=encoding, comment='#')
        
        print(f"📊 Shape: {df_full.shape}")
        print(f"📝 Columns ({len(df_full.columns)}):")
        for i, col in enumerate(df_full.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n🔍 First few rows:")
        print(df.to_string(max_cols=6))
        
        print(f"\n📈 Data types:")
        print(df_full.dtypes)
        
        return df_full
        
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def map_column_names(df, dataset_name):
    """Map various column name formats to standardized feature names."""
    # Standard feature names we want
    standard_features = {
        'pl_orbper': ['pl_orbper', 'Orbital Period [days]', 'koi_period', 'orbital_period'],
        'pl_rade': ['pl_rade', 'Planetary Radius [Earth radii]', 'koi_prad', 'planet_radius'],
        'pl_trandep': ['pl_trandep', 'Transit Depth [ppm]', 'koi_depth', 'transit_depth'],
        'pl_trandur': ['pl_trandur', 'Transit Duration [hours]', 'koi_duration', 'transit_duration'],
        'pl_bmasse': ['pl_bmasse', 'Planet Mass [Earth masses]', 'koi_mass', 'planet_mass'],
        'st_teff': ['st_teff', 'Stellar Effective Temperature [K]', 'koi_seff', 'stellar_temp'],
        'st_rad': ['st_rad', 'Stellar Radius [Solar radii]', 'koi_srad', 'stellar_radius'],
        'sy_dist': ['sy_dist', 'Distance [pc]', 'koi_dist', 'distance']
    }
    
    # Possible label column names
    label_columns = [
        'Disposition Using Kepler Data', 
        'koi_disposition', 
        'tfopwg_disp', 
        'disposition',
        'exoplanet_archive_disposition'
    ]
    
    print(f"\n🔄 Mapping columns for {dataset_name}...")
    
    # Create mapping dictionary
    column_mapping = {}
    found_features = {}
    
    # Find matching columns for features
    for standard_name, possible_names in standard_features.items():
        for possible_name in possible_names:
            if possible_name in df.columns:
                column_mapping[possible_name] = standard_name
                found_features[standard_name] = possible_name
                print(f"  ✅ {standard_name} ← {possible_name}")
                break
    
    # Find label column
    label_col = None
    for col in label_columns:
        if col in df.columns:
            label_col = col
            column_mapping[col] = 'label'
            print(f"  ✅ label ← {label_col}")
            break
    
    if not label_col:
        print(f"  ⚠️ No label column found! Looking for columns containing 'disp'...")
        for col in df.columns:
            if 'disp' in col.lower():
                label_col = col
                column_mapping[col] = 'label'
                print(f"  ✅ label ← {label_col} (found by pattern)")
                break
    
    # Apply mapping
    df_mapped = df.rename(columns=column_mapping)
    
    print(f"  📋 Found {len(found_features)} features out of {len(standard_features)}")
    print(f"  📋 Label column: {label_col if label_col else 'NOT FOUND'}")
    
    return df_mapped, found_features, label_col

def standardize_labels(df, label_col):
    """Standardize label values to CONFIRMED, CANDIDATE, FALSE POSITIVE."""
    if label_col not in df.columns and 'label' not in df.columns:
        print("⚠️ No label column found for standardization")
        return df
    
    label_column = 'label' if 'label' in df.columns else label_col
    
    print(f"\n🏷️ Standardizing labels from column: {label_column}")
    print(f"Original label distribution:")
    print(df[label_column].value_counts())
    
    # Mapping rules for different label formats
    label_mapping = {
        # Kepler/K2 format
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE': 'FALSE POSITIVE',
        # KOI format
        'CONFIRMED PLANET': 'CONFIRMED',
        'PLANET CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE PLANET': 'FALSE POSITIVE',
        # TESS format
        'CP': 'CONFIRMED',  # Confirmed Planet
        'PC': 'CANDIDATE',  # Planet Candidate
        'FP': 'FALSE POSITIVE',  # False Positive
        'FA': 'FALSE POSITIVE',  # False Alarm
    }
    
    # Apply mapping with case insensitive matching
    df['label_standardized'] = df[label_column].astype(str).str.upper().map(
        {k.upper(): v for k, v in label_mapping.items()}
    )
    
    # Fill unmapped values as CANDIDATE (conservative approach)
    df['label_standardized'] = df['label_standardized'].fillna('CANDIDATE')
    
    print(f"\nStandardized label distribution:")
    print(df['label_standardized'].value_counts())
    
    return df

def handle_missing_values_and_outliers(df, features):
    """Handle missing values and remove outliers."""
    print(f"\n🔧 Processing missing values and outliers...")
    
    initial_rows = len(df)
    
    # Check missing values
    missing_info = df[features].isnull().sum()
    print(f"Missing values per feature:")
    for feature, missing_count in missing_info.items():
        if missing_count > 0:
            print(f"  {feature}: {missing_count} ({missing_count/len(df)*100:.1f}%)")
    
    # Fill missing values with median
    for feature in features:
        if df[feature].isnull().sum() > 0:
            median_val = df[feature].median()
            df[feature] = df[feature].fillna(median_val)
            print(f"  ✅ Filled {feature} missing values with median: {median_val:.3f}")
    
    # Remove outliers (beyond 3 standard deviations)
    outlier_mask = pd.Series([True] * len(df))
    
    for feature in features:
        if df[feature].dtype in ['float64', 'int64']:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            
            # Define outlier bounds
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            feature_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
            outlier_mask &= ~feature_outliers
            
            outlier_count = feature_outliers.sum()
            if outlier_count > 0:
                print(f"  📊 {feature}: removed {outlier_count} outliers")
    
    df_clean = df[outlier_mask].copy()
    
    removed_rows = initial_rows - len(df_clean)
    print(f"  🗑️ Removed {removed_rows} rows ({removed_rows/initial_rows*100:.1f}%) with outliers")
    print(f"  ✅ Final dataset: {len(df_clean)} rows")
    
    return df_clean

def main():
    """Main function to process all NASA datasets."""
    print("🚀 NASA Exoplanet Data Processing Pipeline")
    print("="*60)
    
    # Set up paths
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'
    
    # Expected CSV files
    csv_files = {
        'kepler': data_dir / 'kepler_data.csv',
        'k2': data_dir / 'k2_data.csv',
        'tess': data_dir / 'tess_data.csv'
    }
    
    # Check if data files exist
    available_files = {}
    for dataset_name, filepath in csv_files.items():
        if filepath.exists():
            available_files[dataset_name] = filepath
        else:
            print(f"⚠️ Warning: {filepath} not found")
    
    if not available_files:
        print("❌ No data files found! Please ensure CSV files are in the data/ directory")
        return
    
    print(f"Found {len(available_files)} dataset(s): {list(available_files.keys())}")
    
    # Process each dataset
    processed_datasets = []
    all_features = set()
    
    for dataset_name, filepath in available_files.items():
        print(f"\n{'='*40} Processing {dataset_name.upper()} {'='*40}")
        
        # Load and inspect data
        df = inspect_csv_structure(filepath)
        if df is None:
            continue
        
        # Map column names
        df_mapped, found_features, label_col = map_column_names(df, dataset_name)
        
        # Skip if no label column found
        if not label_col and 'label' not in df_mapped.columns:
            print(f"⚠️ Skipping {dataset_name} - no label column found")
            continue
        
        # Standardize labels
        df_standardized = standardize_labels(df_mapped, label_col)
        
        # Add dataset identifier
        df_standardized['dataset_source'] = dataset_name.upper()
        
        # Keep track of available features
        available_features = [f for f in found_features.keys() if f in df_standardized.columns]
        all_features.update(available_features)
        
        print(f"  📋 Available features: {available_features}")
        
        processed_datasets.append(df_standardized)
    
    if not processed_datasets:
        print("❌ No datasets could be processed!")
        return
    
    # Combine all datasets
    print(f"\n{'='*20} COMBINING DATASETS {'='*20}")
    
    # Find common features across all datasets
    common_features = list(all_features)
    print(f"All available features: {common_features}")
    
    # Combine datasets
    combined_df = pd.concat(processed_datasets, ignore_index=True, sort=False)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Dataset distribution:")
    print(combined_df['dataset_source'].value_counts())
    print(f"Label distribution:")
    print(combined_df['label_standardized'].value_counts())
    
    # Select final features and labels
    final_features = [f for f in common_features if f in combined_df.columns]
    final_columns = final_features + ['label_standardized', 'dataset_source']
    
    # Create final dataset with only available features
    final_df = combined_df[final_columns].copy()
    
    # Handle missing values and outliers
    final_df_clean = handle_missing_values_and_outliers(final_df, final_features)
    
    # Save processed data
    output_path = data_dir / 'processed_combined.csv'
    final_df_clean.to_csv(output_path, index=False)
    
    print(f"\n{'='*20} PROCESSING COMPLETE {'='*20}")
    print(f"✅ Processed data saved to: {output_path}")
    print(f"📊 Final dataset shape: {final_df_clean.shape}")
    print(f"🎯 Features: {final_features}")
    print(f"📈 Label distribution:")
    print(final_df_clean['label_standardized'].value_counts())
    
    # Save feature list for training script
    feature_info = {
        'features': final_features,
        'label_column': 'label_standardized',
        'dataset_sources': final_df_clean['dataset_source'].unique().tolist()
    }
    
    import json
    with open(data_dir / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"✅ Feature info saved to: {data_dir / 'feature_info.json'}")
    print("\n🎉 Data processing pipeline completed successfully!")

if __name__ == "__main__":
    main()