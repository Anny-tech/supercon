# %%
"""
Enhanced Feature Engineering for Superconductor Data
====================================================
Author: Ankita Biswas
Project: Dashboard for Superconductors
Date: December 2025

This script improves upon the Kaggle code by:
1. Using our pre-cleaned, high-quality data
2. Better formula normalization handling
3. Enhanced material classification
4. More robust feature generation
5. Proper handling of oxygen variability
"""

# %% [markdown]
# ### Import Packages

# %%
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Configuration

# %%
INPUT_FILE = '/home/digifort/Documents/Data_Management_F25/supercon/clean_data/superconductors_tier2_standard.csv'  # Our cleaned data
OUTPUT_DIR = '/home/digifort/Documents/Data_Management_F25/supercon/feature_engineered_data/'
OUTPUT_FILE = '/home/digifort/Documents/Data_Management_F25/supercon/feature_engineered_data/superconductors_with_features.csv'

print("=" * 70)
print("SUPERCONDUCTOR FEATURE ENGINEERING PIPELINE")
print("=" * 70)

# %% [markdown]
# ### Load data

# %%
print("\n[1/7] Loading cleaned data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} records")

# Convert string representation of sets back to actual sets
df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else set())

print(f"\nInitial columns ({len(df.columns)}): {list(df.columns)}")

# %%
print("\n[2/7] Selecting relevant columns...")

# Keep only columns needed for feature engineering and analysis
columns_to_keep = [
    # Core data
    'data_number',           # Keep for reference
    'chemical_formula',      # Original formula
    'formula_normalized',    # Cleaned formula
    'tc_kelvin',            # Target variable (Tc)
    
    # Composition info
    'elements',             # Set of elements
    'n_elements',           # Number of elements
    
    # Quality flags (useful for filtering later)
    'quality_tier',         # Quality classification
    'has_oxygen_var',       # Oxygen variability flag
    'is_duplicate_formula', # Duplicate flag
    'is_high_tc',          # High-Tc flag
    
    # Material info
    'material_family',      # Cuprate/iron-based/etc.
    'publication_year',     # Discovery year
    
    # Validation info
    'tc_validation',        # Valid/too_low/too_high
]

# Columns to DROP (not needed for feature engineering)
columns_to_drop = [
    'common_formula',       # Redundant with chemical_formula
    'structure_name',       # Not used in features
    'tc_unit',             # Always Kelvin
    'tc_value',            # Replaced by tc_kelvin
    'journal_reference',    # Not used in features
    'missing_formula',      # All False in tier2
    'missing_tc',          # All False in tier2
    'missing_critical',    # All False in tier2
    'tc_has_uncertainty',  # Keep only tc_kelvin
    'tc_validation_reason', # Redundant with tc_validation
    'is_duplicate_exact',  # Keep only is_duplicate_formula
]

# %%
df = df[columns_to_keep]
print(f"Kept {len(columns_to_keep)} relevant columns")
print(f"Dropped {len(columns_to_drop)} redundant columns")

# %%
df

# %% [markdown]
# ### Handling duplicates

# %%

print("\n[3/7] Handling duplicate measurements...")

initial_count = len(df)

# Better approach: use transform and drop_duplicates instead of apply
# First, calculate aggregated Tc values
grouped = df.groupby('formula_normalized')['tc_kelvin'].agg(['mean', 'std', 'count']).reset_index()
grouped.columns = ['formula_normalized', 'tc_kelvin_agg', 'tc_std', 'n_measurements']

# For groups with high variability, remove outliers
for idx, row in grouped.iterrows():
    if row['tc_std'] > 5 and row['n_measurements'] > 1:
        formula = row['formula_normalized']
        formula_data = df[df['formula_normalized'] == formula]['tc_kelvin']
        mean_tc = formula_data.mean()
        std_tc = formula_data.std()
        
        # Remove outliers
        mask = (formula_data >= mean_tc - 2*std_tc) & (formula_data <= mean_tc + 2*std_tc)
        if mask.sum() > 0:
            corrected_mean = formula_data[mask].mean()
            grouped.at[idx, 'tc_kelvin_agg'] = corrected_mean

# Fill NaN std with 0
grouped['tc_std'] = grouped['tc_std'].fillna(0.0)

# Take first occurrence of each formula (to preserve metadata)
df_unique = df.drop_duplicates(subset='formula_normalized', keep='first').copy()

# Merge the aggregated Tc values
df_unique = df_unique.drop(columns=['tc_kelvin'])  # Remove original Tc
df_unique = df_unique.merge(grouped[['formula_normalized', 'tc_kelvin_agg', 'tc_std', 'n_measurements']], 
                             on='formula_normalized', how='left')
df_unique.rename(columns={'tc_kelvin_agg': 'tc_kelvin'}, inplace=True)

print(f"Records before aggregation: {initial_count}")
print(f"Unique formulas after aggregation: {len(df_unique)}")
print(f"Reduced by: {initial_count - len(df_unique)} ({100*(initial_count - len(df_unique))/initial_count:.1f}%)")

# Add measurement count column if not already present
if 'n_measurements' not in df_unique.columns:
    df_unique['n_measurements'] = 1
    df_unique['tc_std'] = 0.0

df = df_unique

# %% [markdown]
# ### Clean formulas for MatMiner

# %%
print("\n[4/7] Preparing formulas for feature generation...")

def clean_formula_for_matminer(formula):
    """
    Clean formula for matminer compatibility:
    - Remove oxygen variability markers (-Y, -Z, -X)
    - Handle variable stoichiometry
    - Keep only valid element symbols and numbers
    """
    if pd.isna(formula):
        return None
    
    import re
    
    # Remove suffixes like -Y, -Z, -X
    formula = re.sub(r'-[XYZ]$', '', formula)
    
    # Remove +X patterns
    formula = re.sub(r'\+[XYZ]', '', formula)
    
    # Remove trailing variable indicators
    formula = formula.rstrip('xyzXYZ')
    
    # Handle some common issues
    # Replace Oz with O (common typo/notation)
    formula = formula.replace('Oz', 'O')
    
    return formula.strip()

df['formula_clean'] = df['formula_normalized'].apply(clean_formula_for_matminer)

# Remove any formulas that couldn't be cleaned
before_clean = len(df)
df = df[df['formula_clean'].notna() & (df['formula_clean'] != '')]
after_clean = len(df)

print(f"Removed {before_clean - after_clean} formulas that couldn't be cleaned")
print(f"Formulas ready for featurization: {len(df)}")

# Show examples
print("\nExample cleaned formulas:")
for i, (orig, clean) in enumerate(zip(df['formula_normalized'].head(5), 
                                       df['formula_clean'].head(5))):
    print(f"  {i+1}. {orig:40s} → {clean}")

# %% [markdown]
# ### Enhanced material classification

# %%
print("\n[5/7] Enhancing material classification...")

def classify_superconductor_enhanced(elements_set):
    """
    Enhanced material classification based on composition.
    More detailed than the original material_family.
    """
    if not elements_set or len(elements_set) == 0:
        return 'Unknown'
    
    # Convert to set if needed
    if isinstance(elements_set, str):
        elements_set = ast.literal_eval(elements_set)
    
    # Alkali metals
    alkali_metals = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'}
    
    # Classification logic (order matters - most specific first)
    
    # Hydrogen-rich (H > 50% atomic fraction would need composition, use presence for now)
    if 'H' in elements_set and len(elements_set) <= 3:
        return 'Hydrogen-rich'
    
    # Organic (alkali + C)
    if any(elem in alkali_metals for elem in elements_set) and 'C' in elements_set:
        return 'Organic'
    
    # Elemental
    if len(elements_set) == 1:
        return 'Elemental'
    
    # Cuprates (Cu + O, most important high-Tc)
    if 'Cu' in elements_set and 'O' in elements_set:
        return 'Cuprate'
    
    # Iron-based
    if 'Fe' in elements_set:
        # Sub-classify iron-based
        if 'As' in elements_set:
            return 'Iron-pnictide'  # FeAs-based
        elif 'Se' in elements_set or 'Te' in elements_set:
            return 'Iron-chalcogenide'  # FeSe/FeTe-based
        else:
            return 'Iron-based'
    
    # Heavy fermion (Ce, U, Pu compounds)
    if any(elem in {'Ce', 'U', 'Pu', 'Np'} for elem in elements_set):
        return 'Heavy-fermion'
    
    # Bismuthates
    if 'Bi' in elements_set and 'O' in elements_set:
        return 'Bismuthate'
    
    # Borocarbides
    if 'B' in elements_set and 'C' in elements_set:
        return 'Borocarbide'
    
    # Niobium compounds
    if 'Nb' in elements_set:
        if 'N' in elements_set:
            return 'Niobium-nitride'
        elif 'Se' in elements_set:
            return 'Niobium-selenide'
        else:
            return 'Niobium-based'
    
    # Mercury-based (often high-Tc)
    if 'Hg' in elements_set and 'O' in elements_set:
        return 'Mercury-cuprate'  # Usually also cuprates
    
    # Magnesium diboride family
    if 'Mg' in elements_set and 'B' in elements_set:
        return 'MgB2-type'
    
    # Ruthenates
    if 'Ru' in elements_set and 'O' in elements_set:
        return 'Ruthenate'
    
    # Cobaltates
    if 'Co' in elements_set and 'O' in elements_set:
        return 'Cobaltate'
    
    # Default
    return 'Other'

df['category_detailed'] = df['elements'].apply(classify_superconductor_enhanced)

print("\nDetailed material categories:")
print(df['category_detailed'].value_counts())



# %% [markdown]
# ### Install and import MatMiner if needed

# %%
print("\n[6/7] Setting up matminer for feature generation...")

try:
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers import composition as cf
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core.composition import Composition
    print("✓ Matminer and pymatgen already installed")
except ImportError:
    print("Installing matminer and pymatgen...")
    import subprocess
    import sys
    
    # Install matminer
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matminer", "-q"])
    
    # Uninstall and reinstall specific pymatgen version
    #subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "pymatgen", "-y", "-q"])
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "pymatgen==2023.3.10", "-q"])
    
    # Import after installation
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers import composition as cf
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core.composition import Composition
    print("Matminer and pymatgen installed successfully")



# %% [markdown]
# ### Generate composition based features using Matminer

# %%
print("\n[7/7] Generating composition-based features...")
print("This may take several minutes...")

# Convert formula strings to Composition objects
print("\n  Converting formulas to compositions...")
df = StrToComposition().featurize_dataframe(df, "formula_clean", ignore_errors=True)

# Count how many failed
n_failed = df['composition'].isna().sum()
if n_failed > 0:
    print(f"  Warning: {n_failed} formulas failed conversion ({100*n_failed/len(df):.1f}%)")
    print(f"  Removing failed conversions...")
    df = df.dropna(subset=['composition'])
    print(f"  Remaining records: {len(df)}")

# Generate features
print("\n  Generating composition features...")
print("  This includes:")
print("    - Stoichiometry features")
print("    - Element property statistics (magpie)")
print("    - Valence orbital properties")
print("    - Ion properties")
print("    - Transition metal fraction")

feature_calculators = MultipleFeaturizer([
    cf.Stoichiometry(),                              # Basic stoichiometry
    cf.ElementProperty.from_preset("magpie"),        # 132 features from magpie
    cf.ValenceOrbital(props=['avg']),               # Valence orbital properties
    cf.IonProperty(fast=True),                      # Ion properties
    cf.TMetalFraction()                             # Transition metal content
])

feature_labels = feature_calculators.feature_labels()
print(f"\n  Generating {len(feature_labels)} features...")

df = feature_calculators.featurize_dataframe(df, col_id='composition', ignore_errors=True)

# Remove any rows where feature generation failed
initial_len = len(df)
df = df.dropna()
final_len = len(df)

if initial_len - final_len > 0:
    print(f"  Removed {initial_len - final_len} records with failed features")

print(f"\n Feature generation complete!")
print(f"  Final dataset: {final_len} records x {len(df.columns)} columns")
print(f"  Features generated: {len(feature_labels)}")

# %% [markdown]
# ### Save data

# %%
print(f"\n[8/8] Saving feature-engineered data...")

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_save_csv(df, filepath):
    """Safely save dataframe, handling sets and complex objects"""
    df_save = df.copy()
    
    # Convert complex types to strings
    for col in df_save.columns:
        if df_save[col].dtype == 'object':
            sample = df_save[col].dropna().iloc[0] if len(df_save[col].dropna()) > 0 else None
            if sample is not None and not isinstance(sample, str):
                df_save[col] = df_save[col].astype(str)
    
    # Try CSV save
    try:
        df_save.to_csv(filepath, index=False)
        return True, "CSV"
    except Exception as e:
        # Fallback to pickle
        pickle_path = filepath.replace('.csv', '.pkl')
        df.to_pickle(pickle_path)
        return True, "PICKLE"

# Save full dataset
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
success, format_type = safe_save_csv(df, output_path)

if format_type == "CSV":
    print(f" Saved CSV: {output_path}")
else:
    pickle_path = output_path.replace('.csv', '.pkl')
    print(f" CSV save failed due to pandas compatibility issue")
    print(f" Saved as pickle instead: {pickle_path}")
    print(f"  To load: df = pd.read_pickle('{pickle_path}')")

# Save feature list
feature_list_path = os.path.join(OUTPUT_DIR, 'feature_list.txt')
with open(feature_list_path, 'w') as f:
    f.write("COMPOSITION-BASED FEATURES\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total features: {len(feature_labels)}\n\n")
    f.write("Feature categories:\n")
    f.write("  1. Stoichiometry features\n")
    f.write("  2. Element property statistics (magpie preset)\n")
    f.write("  3. Valence orbital properties\n")
    f.write("  4. Ion properties\n")
    f.write("  5. Transition metal fraction\n\n")
    f.write("Feature list:\n")
    f.write("-" * 70 + "\n")
    for i, feat in enumerate(feature_labels, 1):
        f.write(f"{i:3d}. {feat}\n")

print(f" Saved: {feature_list_path}")

# Save summary statistics
summary_path = os.path.join(OUTPUT_DIR, 'summary_statistics.txt')
with open(summary_path, 'w') as f:
    f.write("FEATURE ENGINEERING SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Output file: {OUTPUT_FILE}\n\n")
    
    f.write("Data Statistics:\n")
    f.write(f"  Final records: {len(df)}\n")
    f.write(f"  Total columns: {len(df.columns)}\n")
    f.write(f"  Composition features: {len(feature_labels)}\n\n")
    
    f.write("Tc Statistics:\n")
    f.write(f"  Mean: {df['tc_kelvin'].mean():.2f} K\n")
    f.write(f"  Median: {df['tc_kelvin'].median():.2f} K\n")
    f.write(f"  Std: {df['tc_kelvin'].std():.2f} K\n")
    f.write(f"  Min: {df['tc_kelvin'].min():.2f} K\n")
    f.write(f"  Max: {df['tc_kelvin'].max():.2f} K\n\n")
    
    f.write("Material Categories:\n")
    for cat, count in df['category_detailed'].value_counts().items():
        f.write(f"  {cat}: {count} ({100*count/len(df):.1f}%)\n")

print(f" Saved: {summary_path}")


# %% [markdown]
# ### Summary

# %%
print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print(f"\nFinal dataset shape: {df.shape}")
print(f"  Records: {len(df)}")
print(f"  Features: {len(feature_labels)}")
print(f"  Total columns: {len(df.columns)}")

print(f"\nOutput files:")
print(f"  1. {output_path}")
print(f"  2. {feature_list_path}")
print(f"  3. {summary_path}")

print(f"\nTop features by correlation with Tc:")
# Calculate correlations with Tc
correlations = df[feature_labels].corrwith(df['tc_kelvin']).abs().sort_values(ascending=False)
print(correlations.head(10).to_string())

print("\n" + "=" * 70)
print("Next steps:")
print("  1. Explore features in feature_engineered_data/")
print("  2. Build predictive models")
print("  3. Create dashboard with feature-enhanced data")
print("=" * 70)

# %%
df.head(20)

# %%
df.to_csv('fin_v1.csv', index=False)

# %%
  


