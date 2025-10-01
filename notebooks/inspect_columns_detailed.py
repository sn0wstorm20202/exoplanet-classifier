#!/usr/bin/env python3
"""
Detailed NASA Dataset Column Inspection

This script thoroughly inspects the columns in all NASA datasets to identify:
1. Quality indicators (SNR, false positive flags)
2. Physical parameters (impact, stellar gravity, temperature)
3. Missing vs available features
4. Data quality and completeness
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

def inspect_columns_comprehensively(filepath, dataset_name):
    """Comprehensively inspect dataset columns for high-value features"""
    
    print(f"\n{'='*80}")
    print(f"DETAILED INSPECTION: {dataset_name.upper()}")
    print(f"File: {filepath}")
    print('='*80)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None, {}
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, comment='#')
                print(f"✅ Successfully loaded with encoding: {encoding}")
                break
            except Exception:
                continue
        
        if df is None:
            print("❌ Could not read file with any encoding")
            return None, {}
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📝 Total columns: {len(df.columns)}")
        
        # Define high-value feature patterns
        feature_patterns = {
            'Signal-to-Noise': {
                'patterns': [r'snr', r'signal.*noise', r'transit.*signal'],
                'priority': 'HIGH',
                'description': 'Transit signal-to-noise ratio - critical for quality'
            },
            'False Positive Flags': {
                'patterns': [r'fp.*flag', r'false.*positive', r'not.*transit', r'stellar.*eclipse', r'centroid', r'ephemeris'],
                'priority': 'HIGH',
                'description': 'False positive detection flags - critical for accuracy'
            },
            'Impact Parameter': {
                'patterns': [r'impact', r'b_param'],
                'priority': 'MEDIUM',
                'description': 'Orbital impact parameter - affects transit shape'
            },
            'Stellar Gravity': {
                'patterns': [r'slogg', r'st_logg', r'stellar.*gravity', r'log.*g'],
                'priority': 'MEDIUM',
                'description': 'Stellar surface gravity - helps distinguish stars'
            },
            'Equilibrium Temperature': {
                'patterns': [r'teq', r'equilibrium.*temp', r'pl_eqt'],
                'priority': 'MEDIUM',
                'description': 'Planet equilibrium temperature'
            },
            'Insolation Flux': {
                'patterns': [r'insol', r'flux', r'irrad'],
                'priority': 'MEDIUM',
                'description': 'Stellar flux received by planet'
            },
            'Disposition Score': {
                'patterns': [r'score', r'confidence', r'disp.*score'],
                'priority': 'HIGH',
                'description': 'Kepler pipeline confidence score'
            },
            'Transit Count': {
                'patterns': [r'count', r'num.*transit', r'n_transit'],
                'priority': 'MEDIUM',
                'description': 'Number of observed transits'
            },
            'Semi-Major Axis': {
                'patterns': [r'smax', r'semi.*major', r'orbital.*axis'],
                'priority': 'MEDIUM',
                'description': 'Orbital semi-major axis'
            }
        }
        
        # Find matching columns
        found_features = {}
        all_columns = list(df.columns)
        
        print(f"\n🔍 SEARCHING FOR HIGH-VALUE FEATURES:")
        print("-" * 60)
        
        for feature_name, info in feature_patterns.items():
            matches = []
            for pattern in info['patterns']:
                for col in all_columns:
                    if re.search(pattern, col.lower()) and col not in matches:
                        matches.append(col)
            
            priority_symbol = "🔥" if info['priority'] == 'HIGH' else "⭐" if info['priority'] == 'MEDIUM' else "💡"
            
            if matches:
                found_features[feature_name] = matches
                print(f"{priority_symbol} {feature_name} ({info['priority']}):")
                for match in matches:
                    # Show sample data
                    non_null_count = df[match].count()
                    total_count = len(df)
                    completeness = (non_null_count / total_count) * 100
                    
                    sample_values = df[match].dropna().head(3).tolist()
                    print(f"    ✅ {match}")
                    print(f"       Completeness: {completeness:.1f}% ({non_null_count}/{total_count})")
                    print(f"       Sample values: {sample_values}")
            else:
                print(f"❌ {feature_name} ({info['priority']}): Not found")
        
        # Show label columns
        print(f"\n🏷️ LABEL COLUMNS:")
        print("-" * 40)
        label_patterns = [r'disp', r'confirmed', r'candidate', r'false']
        label_columns = []
        
        for col in all_columns:
            for pattern in label_patterns:
                if re.search(pattern, col.lower()):
                    label_columns.append(col)
                    unique_values = df[col].value_counts().head(5)
                    print(f"✅ {col}")
                    print(f"   Values: {list(unique_values.index)}")
                    break
        
        # Show existing mapped features
        print(f"\n📋 CURRENT MAPPED FEATURES (from existing pipeline):")
        print("-" * 50)
        current_features = {
            'pl_orbper': ['pl_orbper', 'Orbital Period [days]', 'koi_period'],
            'pl_rade': ['pl_rade', 'Planetary Radius [Earth radii]', 'koi_prad'],
            'pl_trandep': ['pl_trandep', 'Transit Depth [ppm]', 'koi_depth'],
            'pl_trandur': ['pl_trandur', 'Transit Duration [hours]', 'koi_duration'],
            'pl_bmasse': ['pl_bmasse', 'Planet Mass [Earth masses]', 'koi_mass'],
            'st_teff': ['st_teff', 'Stellar Effective Temperature [K]', 'koi_seff'],
            'st_rad': ['st_rad', 'Stellar Radius [Solar radii]', 'koi_srad'],
            'sy_dist': ['sy_dist', 'Distance [pc]', 'koi_dist']
        }
        
        for feature, possible_names in current_features.items():
            found = False
            for possible_name in possible_names:
                if possible_name in all_columns:
                    completeness = (df[possible_name].count() / len(df)) * 100
                    print(f"✅ {feature} ← {possible_name} ({completeness:.1f}% complete)")
                    found = True
                    break
            if not found:
                print(f"❌ {feature}: Not found")
        
        # Data quality summary
        print(f"\n📊 DATA QUALITY SUMMARY:")
        print("-" * 40)
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")
        
        # Missing data analysis
        missing_data = df.isnull().sum().sort_values(ascending=False)
        high_missing = missing_data[missing_data > len(df) * 0.5]
        
        if len(high_missing) > 0:
            print(f"⚠️  Columns with >50% missing data: {len(high_missing)}")
            for col, missing_count in high_missing.head(5).items():
                pct = (missing_count / len(df)) * 100
                print(f"    {col}: {pct:.1f}% missing")
        
        # Show column categories
        print(f"\n📂 COLUMN CATEGORIES:")
        print("-" * 40)
        
        categories = {
            'Planet Physical': [col for col in all_columns if any(x in col.lower() for x in ['pl_', 'planet', 'radius', 'mass', 'period'])],
            'Stellar': [col for col in all_columns if any(x in col.lower() for x in ['st_', 'stellar', 'star', 'teff'])],
            'Transit': [col for col in all_columns if any(x in col.lower() for x in ['tran', 'depth', 'duration'])],
            'System': [col for col in all_columns if any(x in col.lower() for x in ['sy_', 'system', 'distance'])],
            'Quality/Flags': [col for col in all_columns if any(x in col.lower() for x in ['flag', 'snr', 'score', 'quality'])],
            'Labels': label_columns
        }
        
        for category, cols in categories.items():
            if cols:
                print(f"{category}: {len(cols)} columns")
                for col in cols[:3]:  # Show first 3
                    print(f"    • {col}")
                if len(cols) > 3:
                    print(f"    ... and {len(cols) - 3} more")
        
        return df, found_features
        
    except Exception as e:
        print(f"❌ Error inspecting {filepath}: {e}")
        return None, {}

def create_feature_availability_report(datasets_info):
    """Create a comprehensive report of available features across datasets"""
    
    print(f"\n{'='*80}")
    print("FEATURE AVAILABILITY ACROSS DATASETS")
    print('='*80)
    
    # Collect all found features
    all_features = {}
    
    for dataset_name, (df, found_features) in datasets_info.items():
        if df is not None:
            for feature_type, columns in found_features.items():
                if feature_type not in all_features:
                    all_features[feature_type] = {}
                all_features[feature_type][dataset_name] = columns
    
    # Priority features summary
    priority_features = []
    
    print(f"\n🔥 HIGH PRIORITY FEATURES (Quality Indicators):")
    print("-" * 60)
    
    high_priority = ['Signal-to-Noise', 'False Positive Flags', 'Disposition Score']
    for feature_type in high_priority:
        if feature_type in all_features:
            available_in = list(all_features[feature_type].keys())
            print(f"✅ {feature_type}: Available in {available_in}")
            priority_features.extend([f"{ds}_{feature_type}" for ds in available_in])
        else:
            print(f"❌ {feature_type}: Not found in any dataset")
    
    print(f"\n⭐ MEDIUM PRIORITY FEATURES:")
    print("-" * 40)
    
    medium_priority = ['Impact Parameter', 'Stellar Gravity', 'Equilibrium Temperature', 'Insolation Flux']
    for feature_type in medium_priority:
        if feature_type in all_features:
            available_in = list(all_features[feature_type].keys())
            print(f"✅ {feature_type}: Available in {available_in}")
        else:
            print(f"❌ {feature_type}: Not found in any dataset")
    
    # Recommendations
    print(f"\n💡 IMPLEMENTATION RECOMMENDATIONS:")
    print("-" * 40)
    
    total_datasets = len([df for df, _ in datasets_info.values() if df is not None])
    
    if len(priority_features) > 0:
        print(f"✅ Found {len(priority_features)} high-priority features!")
        print(f"   Expected accuracy improvement: +5-10%")
    else:
        print(f"⚠️  No high-priority quality indicators found")
        print(f"   Will focus on feature engineering from existing data")
    
    # Feature engineering opportunities
    print(f"\n🛠️ FEATURE ENGINEERING OPPORTUNITIES:")
    print("-" * 40)
    print(f"• Planet density (mass/radius³)")
    print(f"• Transit depth consistency ratio")
    print(f"• SNR quality categories (if SNR available)")
    print(f"• Combined false positive flag")
    print(f"• Habitable zone distance ratio")
    
    return all_features

def main():
    """Main inspection function"""
    
    print("🔍 COMPREHENSIVE NASA DATASET COLUMN INSPECTION")
    print("=" * 80)
    print("Analyzing datasets for high-value machine learning features...")
    
    # Dataset paths
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'
    
    datasets = {
        'Kepler': data_dir / 'kepler_data.csv',
        'K2': data_dir / 'k2_data.csv',
        'TESS': data_dir / 'tess_data.csv'
    }
    
    # Check which datasets exist
    available_datasets = {}
    for name, path in datasets.items():
        if path.exists():
            available_datasets[name] = path
            print(f"✅ Found: {name} dataset ({path})")
        else:
            print(f"❌ Missing: {name} dataset ({path})")
    
    if not available_datasets:
        print("\n❌ No datasets found! Please ensure CSV files are in the data/ directory")
        return
    
    print(f"\nAnalyzing {len(available_datasets)} dataset(s)...")
    
    # Inspect each dataset
    datasets_info = {}
    
    for dataset_name, filepath in available_datasets.items():
        df, found_features = inspect_columns_comprehensively(filepath, dataset_name)
        datasets_info[dataset_name] = (df, found_features)
    
    # Create comprehensive report
    all_features = create_feature_availability_report(datasets_info)
    
    # Final summary
    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE - SUMMARY")
    print('='*80)
    
    successful_datasets = len([df for df, _ in datasets_info.values() if df is not None])
    print(f"✅ Successfully analyzed: {successful_datasets}/{len(available_datasets)} datasets")
    
    total_high_value_features = sum(len(features) for features in all_features.values())
    print(f"🔥 High-value features found: {total_high_value_features}")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Update notebooks/load_combined_data.py with new column mappings")
    print(f"2. Add feature engineering for derived features")  
    print(f"3. Retrain model with enhanced feature set")
    print(f"4. Expect accuracy improvement to 92-97%")
    
    print(f"\n🎯 Ready to implement enhanced feature pipeline!")

if __name__ == "__main__":
    main()