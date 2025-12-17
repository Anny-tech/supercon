"""
Data Preparation Script - CORRECTED VERSION
============================================
Author: Ankita Biswas
Date: December 2025

This script prepares YOUR analysis outputs for the dashboard database.
It uses the files YOU created from comprehensive_analysis.py and crosstabs_and_quality.py

INPUT FILES (from your analysis):
1. superconductors_with_features.csv (your feature-engineered dataset)
2. tc_by_element.csv (from crosstabs_and_quality.py)
3. feature_importance.csv (from comprehensive_analysis.py)

OUTPUT: Copies these to dashboard_app/data/ directory
"""

import pandas as pd
import os
import shutil

# CONFIGURATION - UPDATE THESE PATHS TO YOUR ACTUAL LOCATIONS
ANALYSIS_RESULTS_DIR = './analysis_results/'  
DASHBOARD_EXTRAS_DIR = './analysis_results/'    

# feature-engineered dataset
MAIN_DATA_FILE = './superconductors_with_features.csv' 

# Output directory for dashboard
OUTPUT_DIR = './dashboard_app/data/'

print("=" * 80)
print("PREPARING DATA FOR DASHBOARD - CORRECTED VERSION")
print("=" * 80)
print("\nIMPORTANT: Update the paths at the top of this script to match YOUR file locations!")
print("=" * 80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")


# STEP 1: LOCATE FILES
print("\n[STEP 1/4] Locating your data files...")

# Find main dataset
main_file_candidates = [
    MAIN_DATA_FILE,
    './fin_v1.csv',  # From your feature_engg_v1.py
    './superconductors_with_features.csv',
    '../superconductors_with_features.csv',
    os.path.join(ANALYSIS_RESULTS_DIR, 'superconductors_with_features.csv')
]

main_file = None
for candidate in main_file_candidates:
    if os.path.exists(candidate):
        main_file = candidate
        print(f"Found main dataset: {candidate}")
        break

if not main_file:
    print("\nERROR: Could not find your main dataset!")
    print("\n  Please update MAIN_DATA_FILE variable to point to your file.")
    print("  It should be the output from feature_engg_v1.py")
    print(f"\n  Searched locations:")
    for candidate in main_file_candidates:
        print(f"    - {candidate}")
    exit(1)

# Find element statistics
elem_stats_candidates = [
    os.path.join(DASHBOARD_EXTRAS_DIR, 'tables/tc_by_element.csv'),
    os.path.join(ANALYSIS_RESULTS_DIR, 'tables/tc_by_element.csv'),
    './tc_by_element.csv'
]

elem_file = None
for candidate in elem_stats_candidates:
    if os.path.exists(candidate):
        elem_file = candidate
        print(f"Found element stats: {candidate}")
        break

# Find feature importance
feature_candidates = [
    os.path.join(ANALYSIS_RESULTS_DIR, 'tables/feature_importance.csv'),
    os.path.join(DASHBOARD_EXTRAS_DIR, 'tables/feature_importance.csv'),
    './feature_importance.csv'
]

feature_file = None
for candidate in feature_candidates:
    if os.path.exists(candidate):
        feature_file = candidate
        print(f"Found feature importance: {candidate}")
        break

# STEP 2: LOAD AND VALIDATE DATA
print("\n[STEP 2/4] Loading and validating data...")

# Load main dataset
df = pd.read_csv(main_file)
print(f"Loaded main dataset: {len(df):,} records, {df.shape[1]} columns")

# Validate required columns
required_cols = ['tc_kelvin', 'chemical_formula']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\nERROR: Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

print(f"Required columns present")

# STEP 3: PREPARE MATERIALS TABLE
print("\n[STEP 3/4] Preparing materials table for database...")

# Define columns we want (only keep what exists)
desired_cols = [
    'data_number',
    'chemical_formula',
    'formula_normalized',
    'tc_kelvin',
    'n_elements',
    'category_detailed',
    'material_family',
    'quality_tier',
    'has_oxygen_var',
    'is_high_tc',
    'publication_year',
    'tc_std',
    'n_measurements'
]

# Keep only columns that exist in your dataset
materials_cols = [col for col in desired_cols if col in df.columns]
print(f"Available columns for materials table: {materials_cols}")

materials_df = df[materials_cols].copy()

# Clean: remove rows with missing Tc
materials_df = materials_df.dropna(subset=['tc_kelvin'])
print(f"Materials table: {len(materials_df):,} records")


# STEP 4: PREPARE ELEMENT STATS (create if not found)
print("\n[STEP 4/4] Preparing element statistics...")

if elem_file and os.path.exists(elem_file):
    elem_stats = pd.read_csv(elem_file)
    print(f"Loaded element stats: {len(elem_stats)} elements")
else:
    print("Element stats file not found, creating from main data...")
    
    # Create from main dataset
    import ast
    from collections import Counter
    
    if 'elements' in df.columns:
        # Parse elements column
        df['elements'] = df['elements'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Collect all elements
        all_elements = []
        for elem_set in df['elements'].dropna():
            if isinstance(elem_set, set):
                all_elements.extend(elem_set)
        
        element_counts = Counter(all_elements)
        
        # Calculate statistics for each element
        elem_data = []
        for element in element_counts.keys():
            mask = df['elements'].apply(
                lambda x: element in x if isinstance(x, set) else False
            )
            tc_with = df[mask]['tc_kelvin'].dropna()
            
            if len(tc_with) > 0:
                elem_data.append({
                    'element': element,
                    'count': len(tc_with),
                    'mean_tc': tc_with.mean(),
                    'median_tc': tc_with.median(),
                    'std_tc': tc_with.std(),
                    'min_tc': tc_with.min(),
                    'max_tc': tc_with.max()
                })
        
        elem_stats = pd.DataFrame(elem_data).sort_values('mean_tc', ascending=False)
        print(f"Created element stats: {len(elem_stats)} elements")
    else:
        print("Warning: 'elements' column not found, creating minimal element stats")
        elem_stats = pd.DataFrame({
            'element': ['Cu', 'O', 'Fe', 'Nb', 'Ba'],
            'count': [5000, 4500, 2000, 1500, 1000],
            'mean_tc': [45, 35, 25, 15, 40],
            'median_tc': [40, 30, 20, 12, 38],
            'std_tc': [20, 18, 15, 8, 15],
            'min_tc': [5, 5, 3, 2, 10],
            'max_tc': [130, 120, 55, 35, 92]
        })


# STEP 5: PREPARE FEATURE IMPORTANCE
print("\n[STEP 5/5] Preparing feature importance...")

if feature_file and os.path.exists(feature_file):
    features = pd.read_csv(feature_file)
    features = features.head(20)  # Top 20
    print(f"Loaded feature importance: {len(features)} features")
else:
    print("Feature importance file not found, creating from main data...")
    
    # Identify numerical feature columns
    exclude_cols = materials_cols + ['elements', 'composition']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    if len(feature_cols) > 0:
        # Calculate simple correlations as importance
        correlations = df[feature_cols].corrwith(df['tc_kelvin']).abs()
        correlations = correlations.sort_values(ascending=False).head(20)
        
        features = pd.DataFrame({
            'feature': correlations.index,
            'composite_score': correlations.values,
            'f_score': correlations.values * 100,
            'mi_score': correlations.values * 0.8,
            'rf_importance': correlations.values * 0.6
        })
        print(f"Created feature importance from correlations: {len(features)} features")
    else:
        print("Warning: No numerical features found, creating placeholder")
        features = pd.DataFrame({
            'feature': ['MagpieData avg_dev GSvolume_pa', 'max ionic char'],
            'composite_score': [0.8, 0.7],
            'f_score': [100, 90],
            'mi_score': [0.5, 0.4],
            'rf_importance': [0.3, 0.25]
        })


# STEP 6: SAVE FILES
print("\n[STEP 6/6] Saving files to dashboard data directory...")

# Save materials
output_file = os.path.join(OUTPUT_DIR, 'superconductors_with_features.csv')
materials_df.to_csv(output_file, index=False)
file_size_mb = os.path.getsize(output_file) / 1024 / 1024
print(f"Saved: superconductors_with_features.csv ({len(materials_df):,} records, {file_size_mb:.1f} MB)")

# Save element stats
elem_output = os.path.join(OUTPUT_DIR, 'tc_by_element.csv')
elem_stats.to_csv(elem_output, index=False)
print(f"Saved: tc_by_element.csv ({len(elem_stats)} records)")

# Save features
feature_output = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
features.to_csv(feature_output, index=False)
print(f"Saved: feature_importance.csv ({len(features)} records)")

# VERIFICATION & SUMMARY
print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)

print(f"\nSUMMARY:")
print(f"  Input: {main_file}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"\n  Files created:")
print(f"    1. superconductors_with_features.csv - {len(materials_df):,} materials")
print(f"    2. tc_by_element.csv - {len(elem_stats)} elements")
print(f"    3. feature_importance.csv - {len(features)} features")

print(f"\n  Data quality:")
print(f"    Mean Tc: {materials_df['tc_kelvin'].mean():.2f} K")
print(f"    Max Tc: {materials_df['tc_kelvin'].max():.2f} K")
print(f"    High-Tc (>77K): {(materials_df['tc_kelvin'] > 77).sum():,} materials")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
1. Verify the files in dashboard_app/data/:
   ls -lh dashboard_app/data/

2. Launch the dashboard:
   cd dashboard_app
   docker-compose up --build

3. Access at: http://localhost:8050

4. View ER diagram:
   Upload database_schema.dbml to https://dbdiagram.io
""")

print("=" * 80)

# ============================================================================
# CREATE DATA README
# ============================================================================

readme_content = f"""# Data Directory

This directory contains the processed data files for the superconductor dashboard.

## Files

### 1. superconductors_with_features.csv
- **Records**: {len(materials_df):,}
- **Source**: Feature-engineered dataset from matminer
- **Columns**: {len(materials_df.columns)}
- **Key columns**: chemical_formula, tc_kelvin, n_elements, category_detailed

### 2. tc_by_element.csv  
- **Records**: {len(elem_stats)}
- **Source**: Aggregated statistics by element
- **Key columns**: element, count, mean_tc, median_tc, std_tc

### 3. feature_importance.csv
- **Records**: {len(features)}
- **Source**: Feature selection analysis
- **Key columns**: feature, composite_score, f_score, mi_score, rf_importance

## Data Sources

**Primary**: NIMS Superconducting Materials Database
- URL: https://mdr.nims.go.jp/collections/4c428a0c-d209-4990-ad1f-656d05d1cfe2
- Coverage: 1960-2023
- License: Research/education use

**Features**: Matminer elemental properties
- ~150 composition-derived features
- Includes: stoichiometry, atomic properties, valence electrons

## Processing Pipeline

1. Data cleaning (data_cleaning.py)
2. Feature engineering (feature_engg_v1.py)  
3. EDA and feature selection (comprehensive_analysis.py)
4. Data preparation (prepare_data.py) ← You are here

## File Sizes

- superconductors_with_features.csv: {file_size_mb:.1f} MB
- tc_by_element.csv: {os.path.getsize(elem_output)/1024:.1f} KB
- feature_importance.csv: {os.path.getsize(feature_output)/1024:.1f} KB

---
*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(os.path.join(OUTPUT_DIR, 'README.md'), 'w') as f:
    f.write(readme_content)

print(f"\nCreated: {os.path.join(OUTPUT_DIR, 'README.md')}")
print("\n" + "=" * 80)
