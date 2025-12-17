# %% [markdown]
# ### Import Packages

# %%
import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import os

# %% [markdown]
# ### Configurations

# %%
INPUT_FILE = '/home/digifort/Documents/Data_Management_F25/supercon/raw_data/primary.tsv'
OUTPUT_DIR = '/home/digifort/Documents/Data_Management_F25/supercon/clean_data/'
LOG_DIR = '/home/digifort/Documents/Data_Management_F25/supercon/log_files/'

LOG_FILE = os.path.join(LOG_DIR, 'cleaning_log.txt')

# Quality thresholds
MIN_TC = 0.01  
MAX_TC = 300   

# %% [markdown]
# ### Utility Functions

# %%
class CleaningLogger:
    """Log all data cleaning operations for transparency"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.logs = []
    
    def log(self, message):
        self.logs.append(message)
        print(message)
    
    def save(self):
        with open(self.log_file, 'w') as f:
            f.write('\n'.join(self.logs))
        print(f"\nLog saved to {self.log_file}")

# %%
def parse_formula_elements(formula):
    """
    Extract elements from a chemical formula.
    Handles formulas like 'Ba0.2La1.8Cu1O4-Y' 
    Returns set of element symbols
    """
    if pd.isna(formula) or not isinstance(formula, str):
        return set()
    
    # Remove common suffixes
    formula = re.sub(r'-[A-Z]$', '', formula)  # Remove -Y, -Z suffixes
    
    # Pattern to match element symbols (Capital letter followed by optional lowercase)
    element_pattern = r'[A-Z][a-z]?'
    elements = set(re.findall(element_pattern, formula))
    
    # Filter out common non-element symbols
    non_elements = {'Y', 'Z', 'X'}  # These are often used as variables
    # Keep Y as Yttrium is a real element, but in context -Y means oxygen deficiency
    
    return elements

# %%
def normalize_formula(formula):
    """
    Normalize chemical formula by:
    - Converting to standard case
    - Removing extra whitespace
    - Flagging oxygen non-stoichiometry
    """
    if pd.isna(formula) or not isinstance(formula, str):
        return None, True  # Return None and flag as problematic
    
    # Remove extra whitespace
    formula = formula.strip()
    
    # Check for oxygen non-stoichiometry markers
    has_oxygen_var = bool(re.search(r'O\d*-[XYZ]', formula))
    
    return formula, has_oxygen_var

# %%
def extract_tc_value(tc_str):
    """
    Extract numerical Tc value from string.
    Handles various formats and flags uncertainties.
    """
    if pd.isna(tc_str):
        return None, True
    
    # If already numeric
    if isinstance(tc_str, (int, float)):
        return float(tc_str), False
    
    # Convert to string and clean
    tc_str = str(tc_str).strip()
    
    # Check for uncertainty markers
    has_uncertainty = bool(re.search(r'[~<>≈±]', tc_str))
    
    # Extract first number
    match = re.search(r'[-+]?\d*\.?\d+', tc_str)
    if match:
        try:
            value = float(match.group())
            return value, has_uncertainty
        except ValueError:
            return None, True
    
    return None, True

# %%
def validate_tc(tc):
    """
    Validate Tc value is physically reasonable.
    Returns validation status and reason.
    """
    if pd.isna(tc):
        return 'missing', 'Tc value missing'
    
    if tc < MIN_TC:
        return 'too_low', f'Tc below threshold ({MIN_TC}K)'
    
    if tc > MAX_TC:
        return 'too_high', f'Tc above threshold ({MAX_TC}K)'
    
    return 'valid', 'Valid Tc'

# %% [markdown]
# ### Cleaning

# %%
def load_data(file_path, logger):
    """Load the raw TSV data"""
    logger.log("=" * 70)
    logger.log("STEP 1: LOADING DATA")
    logger.log("=" * 70)
    
    # The first 3 rows contain metadata/headers
    # Row 0: column numbers
    # Row 1: descriptive column names
    # Row 2: short column codes
    # Actual data starts from row 3
    
    df = pd.read_csv(file_path, sep='\t', skiprows=3, encoding='utf-8')
    
    logger.log(f"Loaded {len(df)} records from {file_path}")
    logger.log(f"Columns: {list(df.columns)}")
    
    return df

# %%
def standardize_columns(df, logger):
    """Rename columns to standardized names"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 2: STANDARDIZING COLUMN NAMES")
    logger.log("=" * 70)
    
    # Based on the header structure observed
    column_mapping = {
        df.columns[0]: 'data_number',
        df.columns[1]: 'common_formula',
        df.columns[2]: 'chemical_formula',
        df.columns[3]: 'structure_name',
        df.columns[4]: 'tc_unit',
        df.columns[5]: 'tc_value',
        df.columns[6]: 'journal_reference'
    }
    
    df = df.rename(columns=column_mapping)
    logger.log(f"Standardized column names: {list(df.columns)}")
    
    return df

# %%
def handle_missing_values(df, logger):
    """Identify and flag missing values"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 3: HANDLING MISSING VALUES")
    logger.log("=" * 70)
    
    # Count missing values per column
    missing_counts = df.isnull().sum()
    logger.log("\nMissing value counts by column:")
    for col, count in missing_counts.items():
        pct = 100 * count / len(df)
        logger.log(f"  {col}: {count} ({pct:.2f}%)")
    
    # Flag rows with critical missing data
    df['missing_formula'] = df['chemical_formula'].isnull()
    df['missing_tc'] = df['tc_value'].isnull()
    df['missing_critical'] = df['missing_formula'] | df['missing_tc']
    
    n_critical = df['missing_critical'].sum()
    logger.log(f"\nRows with critical missing data (formula or Tc): {n_critical}")
    
    return df

# %%
def clean_formulas(df, logger):
    """Clean and normalize chemical formulas"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 4: CLEANING CHEMICAL FORMULAS")
    logger.log("=" * 70)
    
    # Apply normalization
    results = df['chemical_formula'].apply(normalize_formula)
    df['formula_normalized'] = [r[0] for r in results]
    df['has_oxygen_var'] = [r[1] for r in results]
    
    # Extract elements
    df['elements'] = df['formula_normalized'].apply(parse_formula_elements)
    df['n_elements'] = df['elements'].apply(len)
    
    logger.log(f"Formulas with oxygen non-stoichiometry: {df['has_oxygen_var'].sum()}")
    logger.log(f"\nElement count distribution:")
    logger.log(df['n_elements'].value_counts().sort_index().to_string())
    
    # Identify most common elements
    all_elements = []
    for elem_set in df['elements'].dropna():
        all_elements.extend(elem_set)
    
    element_counts = Counter(all_elements)
    logger.log(f"\nTop 15 most common elements:")
    for elem, count in element_counts.most_common(15):
        pct = 100 * count / len(df)
        logger.log(f"  {elem}: {count} ({pct:.1f}%)")
    
    return df

# %%

def clean_tc_values(df, logger):
    """Clean and validate Tc values"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 5: CLEANING Tc VALUES")
    logger.log("=" * 70)
    
    # Extract Tc values
    results = df['tc_value'].apply(extract_tc_value)
    df['tc_kelvin'] = [r[0] for r in results]
    df['tc_has_uncertainty'] = [r[1] for r in results]
    
    # Validate Tc values
    validation_results = df['tc_kelvin'].apply(validate_tc)
    df['tc_validation'] = [r[0] for r in validation_results]
    df['tc_validation_reason'] = [r[1] for r in validation_results]
    
    # Summary statistics
    logger.log("\nTc Validation Results:")
    logger.log(df['tc_validation'].value_counts().to_string())
    
    valid_tcs = df[df['tc_validation'] == 'valid']['tc_kelvin']
    logger.log(f"\nValid Tc Statistics:")
    logger.log(f"  Count: {len(valid_tcs)}")
    logger.log(f"  Mean: {valid_tcs.mean():.2f} K")
    logger.log(f"  Median: {valid_tcs.median():.2f} K")
    logger.log(f"  Std: {valid_tcs.std():.2f} K")
    logger.log(f"  Min: {valid_tcs.min():.2f} K")
    logger.log(f"  Max: {valid_tcs.max():.2f} K")
    logger.log(f"  25th percentile: {valid_tcs.quantile(0.25):.2f} K")
    logger.log(f"  75th percentile: {valid_tcs.quantile(0.75):.2f} K")
    
    return df

# %%
def detect_duplicates(df, logger):
    """Detect potential duplicate entries"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 6: DETECTING DUPLICATES")
    logger.log("=" * 70)
    
    # Check for exact formula duplicates
    formula_dups = df['formula_normalized'].duplicated(keep=False)
    n_formula_dups = formula_dups.sum()
    
    logger.log(f"Records with duplicate formulas: {n_formula_dups}")
    
    # Check for exact formula + Tc duplicates
    formula_tc_dups = df.duplicated(subset=['formula_normalized', 'tc_kelvin'], keep=False)
    n_exact_dups = formula_tc_dups.sum()
    
    logger.log(f"Records with duplicate formula + Tc: {n_exact_dups}")
    
    df['is_duplicate_formula'] = formula_dups
    df['is_duplicate_exact'] = formula_tc_dups
    
    # Show examples of duplicates
    if n_formula_dups > 0:
        logger.log("\nExample duplicate formulas (first 5):")
        dup_formulas = df[formula_dups].groupby('formula_normalized').head(2)
        for formula, group in dup_formulas.groupby('formula_normalized'):
            if len(group) > 1:
                logger.log(f"\n  Formula: {formula}")
                for _, row in group.iterrows():
                    logger.log(f"    Data #{row['data_number']}: Tc = {row['tc_kelvin']} K")
                break  # Just show first example
    
    return df



# %%
def create_quality_tiers(df, logger):
    """Create quality tier classifications"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 7: CREATING QUALITY TIERS")
    logger.log("=" * 70)
    
    # Define quality tiers
    # Tier 1 (Strict): Valid Tc, no missing critical data, no oxygen var, not duplicate
    tier1 = (
        (df['tc_validation'] == 'valid') &
        (~df['missing_critical']) &
        (~df['has_oxygen_var']) &
        (~df['is_duplicate_exact'])
    )
    
    # Tier 2 (Standard): Valid Tc, no missing critical data
    tier2 = (
        (df['tc_validation'] == 'valid') &
        (~df['missing_critical'])
    )
    
    # Tier 3 (Inclusive): Has Tc value (even if questionable)
    tier3 = df['tc_kelvin'].notna()
    
    df['quality_tier'] = 'excluded'
    df.loc[tier3, 'quality_tier'] = 'tier3_inclusive'
    df.loc[tier2, 'quality_tier'] = 'tier2_standard'
    df.loc[tier1, 'quality_tier'] = 'tier1_strict'
    
    logger.log("\nQuality Tier Distribution:")
    logger.log(df['quality_tier'].value_counts().to_string())
    
    # Calculate tier percentages
    for tier in ['tier1_strict', 'tier2_standard', 'tier3_inclusive']:
        count = (df['quality_tier'] == tier).sum()
        pct = 100 * count / len(df)
        logger.log(f"  {tier}: {count} records ({pct:.1f}%)")
    
    return df

# %%
def add_metadata(df, logger):
    """Add useful metadata columns"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 8: ADDING METADATA")
    logger.log("=" * 70)
    
    # Extract year from journal reference (if possible)
    def extract_year(ref):
        if pd.isna(ref):
            return None
        match = re.search(r'\((\d{4})\)', str(ref))
        if match:
            return int(match.group(1))
        match = re.search(r'(\d{4})', str(ref))
        if match:
            return int(match.group(1))
        return None
    
    df['publication_year'] = df['journal_reference'].apply(extract_year)
    
    year_counts = df['publication_year'].value_counts().sort_index()
    logger.log(f"\nPublications by year: {len(year_counts)} unique years")
    if len(year_counts) > 0:
        logger.log(f"  Earliest: {year_counts.index.min()}")
        logger.log(f"  Latest: {year_counts.index.max()}")
    
    # Flag high-Tc materials (>77K, liquid nitrogen temperature)
    df['is_high_tc'] = df['tc_kelvin'] > 77
    
    # Categorize by common superconductor families
    def categorize_family(elements):
        if pd.isna(elements) or len(elements) == 0:
            return 'unknown'
        
        if 'Cu' in elements and 'O' in elements:
            return 'cuprate'
        elif 'Fe' in elements:
            return 'iron_based'
        elif 'Nb' in elements:
            return 'niobium'
        elif 'Hg' in elements:
            return 'mercury_based'
        else:
            return 'other'
    
    df['material_family'] = df['elements'].apply(categorize_family)
    
    logger.log("\nMaterial Family Distribution:")
    logger.log(df['material_family'].value_counts().to_string())
    
    return df



# %%
def save_cleaned_data(df, output_dir, logger):
    """Save cleaned data in multiple formats"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 9: SAVING CLEANED DATA")
    logger.log("=" * 70)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset with all flags
    full_path = os.path.join(output_dir, 'superconductors_full_cleaned.csv')
    df.to_csv(full_path, index=False)
    logger.log(f"Saved full dataset: {full_path}")
    
    # Save tier 1 (strict quality)
    tier1_df = df[df['quality_tier'] == 'tier1_strict'].copy()
    tier1_path = os.path.join(output_dir, 'superconductors_tier1_strict.csv')
    tier1_df.to_csv(tier1_path, index=False)
    logger.log(f"Saved Tier 1 dataset: {tier1_path} ({len(tier1_df)} records)")
    
    # Save tier 2 (standard quality)
    tier2_df = df[df['quality_tier'].isin(['tier1_strict', 'tier2_standard'])].copy()
    tier2_path = os.path.join(output_dir, 'superconductors_tier2_standard.csv')
    tier2_df.to_csv(tier2_path, index=False)
    logger.log(f"Saved Tier 2 dataset: {tier2_path} ({len(tier2_df)} records)")
    
    # Create summary statistics file
    summary_path = os.path.join(output_dir, 'data_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SUPERCONDUCTOR DATA CLEANING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Tier 1 (strict): {len(tier1_df)} ({100*len(tier1_df)/len(df):.1f}%)\n")
        f.write(f"Tier 2 (standard): {len(tier2_df)} ({100*len(tier2_df)/len(df):.1f}%)\n\n")
        
        f.write("Tc Statistics (valid records):\n")
        valid_tc = df[df['tc_validation'] == 'valid']['tc_kelvin']
        f.write(f"  Mean: {valid_tc.mean():.2f} K\n")
        f.write(f"  Median: {valid_tc.median():.2f} K\n")
        f.write(f"  Range: {valid_tc.min():.2f} - {valid_tc.max():.2f} K\n\n")
        
        f.write("Most common elements:\n")
        all_elements = []
        for elem_set in df['elements'].dropna():
            all_elements.extend(elem_set)
        for elem, count in Counter(all_elements).most_common(10):
            f.write(f"  {elem}: {count}\n")
    
    logger.log(f"Saved summary: {summary_path}")
    
    return df

# %%
def generate_quality_report(df, logger):
    """Generate detailed quality report"""
    logger.log("\n" + "=" * 70)
    logger.log("STEP 10: QUALITY REPORT")
    logger.log("=" * 70)
    
    logger.log("\nDATA QUALITY SUMMARY:")
    logger.log(f"  Total records: {len(df)}")
    logger.log(f"  Records with valid Tc: {(df['tc_validation'] == 'valid').sum()}")
    logger.log(f"  Records with oxygen variability: {df['has_oxygen_var'].sum()}")
    logger.log(f"  Duplicate formulas: {df['is_duplicate_formula'].sum()}")
    logger.log(f"  High-Tc materials (>77K): {df['is_high_tc'].sum()}")
    
    logger.log("\nRECOMMENDATIONS:")
    logger.log("  1. Use 'tier1_strict' for high-quality analyses")
    logger.log("  2. Use 'tier2_standard' for broader coverage")
    logger.log("  3. Filter by 'has_oxygen_var' when precise stoichiometry matters")
    logger.log("  4. Check 'is_duplicate_formula' flag for multiple measurements")



# %%
def main():
    """Run the complete data cleaning pipeline"""
    
    # Initialize logger
    logger = CleaningLogger(LOG_FILE)
    logger.log("SUPERCONDUCTOR DATA CLEANING PIPELINE")
    logger.log(f"Date: {pd.Timestamp.now()}")
    logger.log(f"Input: {INPUT_FILE}")
    
    try:
        # Execute pipeline steps
        df = load_data(INPUT_FILE, logger)
        df = standardize_columns(df, logger)
        df = handle_missing_values(df, logger)
        df = clean_formulas(df, logger)
        df = clean_tc_values(df, logger)
        df = detect_duplicates(df, logger)
        df = create_quality_tiers(df, logger)
        df = add_metadata(df, logger)
        df = save_cleaned_data(df, OUTPUT_DIR, logger)
        generate_quality_report(df, logger)
        
        logger.log("\n" + "=" * 70)
        logger.log("CLEANING PIPELINE COMPLETED SUCCESSFULLY")
        logger.log("=" * 70)
        
    except Exception as e:
        logger.log(f"\nERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
    
    finally:
        logger.save()

# %%
if __name__ == "__main__":
    main()

# %%



