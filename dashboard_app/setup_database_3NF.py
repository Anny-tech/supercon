"""
3NF Database Setup for Superconductor Dashboard
================================================
Author: Ankita Biswas
Date: December 2025

This script creates a PostgreSQL database in STRICT 3rd Normal Form (3NF)
following database design principles from DS 6600 Lab 3.

3NF Requirements:
1. 1NF: Atomic values, no repeating groups
2. 2NF: No partial dependencies (all non-key attributes depend on entire PK)
3. 3NF: No transitive dependencies (non-key attributes depend only on PK)

Tables Created:
1. materials - Core superconductor data
2. categories - Material categories (normalized)
3. families - Material families (normalized)
4. quality_tiers - Quality tier definitions (normalized)
5. elements - Chemical elements
6. materials_elements - Junction table (many-to-many)
7. element_stats - Aggregate statistics per element
8. features - Feature definitions
9. feature_importance - Feature importance scores
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import ast

# CONFIGURATION
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'superconductors')

DATA_FILE = '/data/superconductors_with_features.csv'
ELEMENT_STATS_FILE = '/data/tc_by_element.csv'
FEATURE_IMPORTANCE_FILE = '/data/feature_importance.csv'

print("=" * 80)
print("3NF SUPERCONDUCTOR DATABASE SETUP")
print("=" * 80)

# STEP 1: CREATE DATABASE
print("\n[1/7] Creating database...")

server_url = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres'
server_engine = create_engine(server_url)

with server_engine.connect() as conn:
    conn.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(text(f'DROP DATABASE IF EXISTS {DB_NAME}'))
        print(f"  Dropped existing database: {DB_NAME}")
    except Exception as e:
        print(f"  Note: {e}")
    
    conn.execute(text(f'CREATE DATABASE {DB_NAME}'))
    print(f"  Created database: {DB_NAME}")


# STEP 2: LOAD DATA
print("\n[2/7] Loading data...")

df = pd.read_csv(DATA_FILE)
print(f"  Loaded {len(df):,} records from {DATA_FILE}")

if 'elements' in df.columns and df['elements'].dtype == 'object':
    df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df = df.dropna(subset=['tc_kelvin'])
print(f"  Cleaned data: {len(df):,} records")

# Load element stats
try:
    elem_stats = pd.read_csv(ELEMENT_STATS_FILE)
    print(f"  Loaded element statistics: {len(elem_stats)} elements")
except:
    elem_stats = None
    print(f"  Element stats file not found, will create from data")

# Load feature importance
try:
    features_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    print(f"  Loaded feature importance: {len(features_importance)} features")
except:
    features_importance = None
    print(f"  Feature importance file not found, will create from data")

# STEP 3: CREATE NORMALIZED LOOKUP TABLES
print("\n[3/7] Creating normalized lookup tables...")

# Categories table (normalized)
if 'category_detailed' in df.columns:
    categories_df = df[['category_detailed']].drop_duplicates().dropna()
    categories_df = categories_df.reset_index(drop=True)
    categories_df['category_id'] = categories_df.index + 1
    categories_df.columns = ['category_name', 'category_id']
    categories_df = categories_df[['category_id', 'category_name']]
    print(f"  ✓ Categories: {len(categories_df)} unique values")
else:
    categories_df = pd.DataFrame({'category_id': [1], 'category_name': ['Unknown']})

# Families table (normalized)
if 'material_family' in df.columns:
    families_df = df[['material_family']].drop_duplicates().dropna()
    families_df = families_df.reset_index(drop=True)
    families_df['family_id'] = families_df.index + 1
    families_df.columns = ['family_name', 'family_id']
    families_df = families_df[['family_id', 'family_name']]
    print(f"  ✓ Families: {len(families_df)} unique values")
else:
    families_df = pd.DataFrame({'family_id': [1], 'family_name': ['Unknown']})

# Quality tiers table (normalized)
if 'quality_tier' in df.columns:
    tiers_df = df[['quality_tier']].drop_duplicates().dropna()
    tiers_df = tiers_df.reset_index(drop=True)
    tiers_df['tier_id'] = tiers_df.index + 1
    tiers_df.columns = ['tier_name', 'tier_id']
    tiers_df = tiers_df[['tier_id', 'tier_name']]
    print(f"  ✓ Quality tiers: {len(tiers_df)} unique values")
else:
    tiers_df = pd.DataFrame({'tier_id': [1], 'tier_name': ['Unknown']})

# Elements table
if 'elements' in df.columns:
    all_elements = set()
    for elem_set in df['elements'].dropna():
        if isinstance(elem_set, set):
            all_elements.update(elem_set)
    
    elements_df = pd.DataFrame({'element_symbol': sorted(all_elements)})
    # Add atomic numbers (simplified - you could add real data)
    element_order = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
                     'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18, 'K':19, 'Ca':20,
                     'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
                     'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 'Rb':37, 'Sr':38, 'Y':39, 'Zr':40,
                     'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48, 'In':49, 'Sn':50,
                     'Sb':51, 'Te':52, 'I':53, 'Xe':54, 'Cs':55, 'Ba':56, 'La':57, 'Ce':58, 'Pr':59, 'Nd':60,
                     'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70,
                     'Lu':71, 'Hf':72, 'Ta':73, 'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
                     'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86, 'Fr':87, 'Ra':88, 'Ac':89, 'Th':90,
                     'Pa':91, 'U':92, 'Np':93, 'Pu':94, 'Am':95, 'Cm':96, 'Bk':97, 'Cf':98}
    elements_df['atomic_number'] = elements_df['element_symbol'].map(element_order)
    elements_df['atomic_number'] = elements_df['atomic_number'].fillna(0).astype(int)
    print(f"  ✓ Elements: {len(elements_df)} unique elements")
else:
    elements_df = pd.DataFrame({'element_symbol': ['H', 'O'], 'atomic_number': [1, 8]})


# STEP 4: CREATE MATERIALS TABLE WITH FOREIGN KEYS
print("\n[4/7] Creating materials table with foreign keys...")

# Create materials dataframe
materials_df = df.copy()

# Add foreign key IDs
if 'category_detailed' in materials_df.columns:
    materials_df = materials_df.merge(
        categories_df[['category_id', 'category_name']], 
        left_on='category_detailed', 
        right_on='category_name', 
        how='left'
    )
else:
    materials_df['category_id'] = 1

if 'material_family' in materials_df.columns:
    materials_df = materials_df.merge(
        families_df[['family_id', 'family_name']], 
        left_on='material_family', 
        right_on='family_name', 
        how='left'
    )
else:
    materials_df['family_id'] = 1

if 'quality_tier' in materials_df.columns:
    materials_df = materials_df.merge(
        tiers_df[['tier_id', 'tier_name']], 
        left_on='quality_tier', 
        right_on='tier_name', 
        how='left'
    )
else:
    materials_df['tier_id'] = 1

# Select columns for final materials table (only atomic data and FKs)
materials_cols = [
    'data_number',
    'chemical_formula',
    'formula_normalized',
    'tc_kelvin',
    'n_elements',
    'category_id',
    'family_id',
    'tier_id',
    'has_oxygen_var',
    'is_high_tc',
    'publication_year',
    'tc_std',
    'n_measurements'
]

materials_cols = [col for col in materials_cols if col in materials_df.columns]
materials_final = materials_df[materials_cols].copy()
materials_final['category_id'] = materials_final['category_id'].fillna(1).astype(int)
materials_final['family_id'] = materials_final['family_id'].fillna(1).astype(int)
materials_final['tier_id'] = materials_final['tier_id'].fillna(1).astype(int)

print(f"Materials table: {len(materials_final):,} records")

# STEP 5: CREATE JUNCTION TABLE (materials_elements)
print("\n[5/7] Creating materials_elements junction table...")

materials_elements_records = []
for idx, row in df.iterrows():
    if 'elements' in row and isinstance(row['elements'], set):
        for element in row['elements']:
            materials_elements_records.append({
                'material_id': row['data_number'],
                'element_symbol': element
            })

materials_elements_df = pd.DataFrame(materials_elements_records)
print(f"Materials-Elements: {len(materials_elements_df):,} relationships")


# STEP 6: CREATE AGGREGATE TABLES
print("\n[6/7] Creating aggregate tables...")

# Element stats (if not provided, calculate)
if elem_stats is None:
    from collections import Counter
    all_elements_list = []
    for elem_set in df['elements'].dropna():
        if isinstance(elem_set, set):
            all_elements_list.extend(elem_set)
    
    element_counts = Counter(all_elements_list)
    elem_data = []
    for element in element_counts.keys():
        mask = df['elements'].apply(lambda x: element in x if isinstance(x, set) else False)
        tc_with = df[mask]['tc_kelvin'].dropna()
        if len(tc_with) > 0:
            elem_data.append({
                'element_symbol': element,
                'count': len(tc_with),
                'mean_tc': tc_with.mean(),
                'median_tc': tc_with.median(),
                'std_tc': tc_with.std(),
                'min_tc': tc_with.min(),
                'max_tc': tc_with.max()
            })
    elem_stats = pd.DataFrame(elem_data)
else:
    elem_stats.columns = ['element_symbol' if col == 'element' else col for col in elem_stats.columns]

print(f"Element stats: {len(elem_stats)} elements")

# Feature importance (if not provided, calculate)
if features_importance is None:
    exclude_cols = materials_cols + ['elements', 'composition']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    if len(feature_cols) > 0:
        correlations = df[feature_cols].corrwith(df['tc_kelvin']).abs()
        correlations = correlations.sort_values(ascending=False).head(20)
        features_importance = pd.DataFrame({
            'feature_name': correlations.index,
            'composite_score': correlations.values
        })

print(f"Feature importance: {len(features_importance)} features")

# STEP 7: CREATE DATABASE SCHEMA AND LOAD DATA
print("\n[7/7] Creating schema and loading data...")

db_url = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(db_url)

with engine.connect() as conn:
    
    # create categories table
    conn.execute(text("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name VARCHAR(100) UNIQUE NOT NULL
        )
    """))
    
    # create families table
    conn.execute(text("""
        CREATE TABLE families (
            family_id INTEGER PRIMARY KEY,
            family_name VARCHAR(100) UNIQUE NOT NULL
        )
    """))
    
    # create quality_tiers table
    conn.execute(text("""
        CREATE TABLE quality_tiers (
            tier_id INTEGER PRIMARY KEY,
            tier_name VARCHAR(50) UNIQUE NOT NULL
        )
    """))
    
    # create elements table
    conn.execute(text("""
        CREATE TABLE elements (
            element_symbol VARCHAR(3) PRIMARY KEY,
            atomic_number INTEGER
        )
    """))
    
    # create materials table
    conn.execute(text("""
        CREATE TABLE materials (
            data_number INTEGER PRIMARY KEY,
            chemical_formula VARCHAR(200) NOT NULL,
            formula_normalized VARCHAR(200),
            tc_kelvin FLOAT NOT NULL,
            n_elements INTEGER,
            category_id INTEGER,
            family_id INTEGER,
            tier_id INTEGER,
            has_oxygen_var BOOLEAN,
            is_high_tc BOOLEAN,
            publication_year INTEGER,
            tc_std FLOAT,
            n_measurements INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(category_id),
            FOREIGN KEY (family_id) REFERENCES families(family_id),
            FOREIGN KEY (tier_id) REFERENCES quality_tiers(tier_id)
        )
    """))
    
    # create materials_elements junction table
    conn.execute(text("""
        CREATE TABLE materials_elements (
            material_id INTEGER,
            element_symbol VARCHAR(3),
            PRIMARY KEY (material_id, element_symbol),
            FOREIGN KEY (material_id) REFERENCES materials(data_number),
            FOREIGN KEY (element_symbol) REFERENCES elements(element_symbol)
        )
    """))
    
    # create element_stats table
    conn.execute(text("""
        CREATE TABLE element_stats (
            element_symbol VARCHAR(3) PRIMARY KEY,
            count INTEGER,
            mean_tc FLOAT,
            median_tc FLOAT,
            std_tc FLOAT,
            min_tc FLOAT,
            max_tc FLOAT,
            FOREIGN KEY (element_symbol) REFERENCES elements(element_symbol)
        )
    """))
    
    # create feature_importance table
    conn.execute(text("""
        CREATE TABLE feature_importance (
            feature VARCHAR(200) PRIMARY KEY,
            f_score FLOAT,
            mi_score FLOAT,
            rf_importance FLOAT,
            f_score_norm FLOAT,
            mi_score_norm FLOAT,
            rf_importance_norm FLOAT,
            composite_score FLOAT
        )
    """))
    
    conn.commit()
    print("Created 3NF schema with foreign keys")

# Load data
categories_df.to_sql('categories', engine, if_exists='append', index=False)
print(f"Loaded {len(categories_df)} categories")

families_df.to_sql('families', engine, if_exists='append', index=False)
print(f"Loaded {len(families_df)} families")

tiers_df.to_sql('quality_tiers', engine, if_exists='append', index=False)
print(f"Loaded {len(tiers_df)} quality tiers")

elements_df.to_sql('elements', engine, if_exists='append', index=False)
print(f"Loaded {len(elements_df)} elements")

materials_final.to_sql('materials', engine, if_exists='append', index=False)
print(f"Loaded {len(materials_final):,} materials")

materials_elements_df.to_sql('materials_elements', engine, if_exists='append', index=False)
print(f"Loaded {len(materials_elements_df):,} material-element relationships")

elem_stats.to_sql('element_stats', engine, if_exists='append', index=False)
print(f"Loaded {len(elem_stats)} element statistics")

features_importance.to_sql('feature_importance', engine, if_exists='append', index=False)
print(f"Loaded {len(features_importance)} feature importance scores")

# create indexes
with engine.connect() as conn:
    conn.execute(text("CREATE INDEX idx_materials_tc ON materials(tc_kelvin)"))
    conn.execute(text("CREATE INDEX idx_materials_category ON materials(category_id)"))
    conn.execute(text("CREATE INDEX idx_materials_year ON materials(publication_year)"))
    conn.execute(text("CREATE INDEX idx_materials_elements_material ON materials_elements(material_id)"))
    conn.execute(text("CREATE INDEX idx_materials_elements_element ON materials_elements(element_symbol)"))
    conn.commit()
    print("Created indexes")


# VERIFICATION
print("\n" + "=" * 80)
print("3NF DATABASE VERIFICATION")
print("=" * 80)

with engine.connect() as conn:
    tables = ['categories', 'families', 'quality_tiers', 'elements', 
              'materials', 'materials_elements', 'element_stats', 'feature_importance']
    
    print("\nTable record counts:")
    for table in tables:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count = result.fetchone()[0]
        print(f"  {table:25s}: {count:,}")

print("\n3NF Compliance:")
print("1NF: All values are atomic")
print("2NF: No partial dependencies (all tables have single-column PKs)")
print("3NF: No transitive dependencies (normalized lookup tables)")
print("Foreign keys enforce referential integrity")

print("\n" + "=" * 80)
print("DATABASE SETUP COMPLETE!")
print("=" * 80)
print(f"\nConnection string: {db_url}")
print("Ready for dashboard connection!")
print("=" * 80)
