# Superconductor Data Pipeline

> **Clean, explore, and visualize 26,000+ superconductor records from the NIMS database**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Data Quality](https://img.shields.io/badge/Data%20Quality-99.6%25-success.svg)]()
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

A reproducible data cleaning and visualization pipeline for promoting and pushing research on superconducturs a fascinating material system. 

---

## What This Does

Takes messy superconductor data → Cleans it rigorously → Provides analysis-ready datasets

**Input:** 26,357 raw records from NIMS database  
**Output:** 26,248 validated records (99.6% quality) + visualizations + interactive tools

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/superconductor-data-pipeline.git
pip install pandas numpy matplotlib seaborn scipy

# Run pipeline
python superconductor_data_cleaning.py

# Explore data
python data_explorer.py
```

```python
# Load cleaned data
import pandas as pd
df = pd.read_csv('cleaned_data/superconductors_tier2_standard.csv')

# Find high-Tc cuprates
high_tc = df[(df['tc_kelvin'] > 77) & (df['material_family'] == 'cuprate')]
print(f"Found {len(high_tc)} high-Tc cuprates")
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Total Materials | 26,357 |
| Valid Records | 26,248 (99.6%) |
| Mean Tc | 32.78 K |
| High-Tc (>77K) | 4,707 (17.9%) |
| Elements | 92 unique |
| Time Span | 1911-2025 |

### Material Families
- **Cuprates:** 41% (highest Tc ~60K avg)
- **Iron-based:** 9.4%
- **Other:** 49.6%

### Top Elements
O (47.6%) · Cu (46.5%) · Ba (29.2%) · Sr (21.0%) · Y (18.3%)

---

## What You Get

### Cleaned Datasets (3 quality tiers)
```
cleaned_data/
├── superconductors_tier1_strict.csv      # Publication quality (21,580)
├── superconductors_tier2_standard.csv    # Standard quality (26,248) ⭐ Recommended
└── superconductors_full_cleaned.csv      # Everything (26,357)
```

### Python Scripts
- `superconductor_data_cleaning.py` - Main pipeline
- `superconductor_visualization.py` - Generate plots
- `data_explorer.py` - Interactive queries

### Visualizations
![Preview](cleaned_data/figures/tc_distributions.png)

See `cleaned_data/figures/` for all visualizations.

---

## Pipeline Features

### Data Cleaning
✅ Formula parsing & element extraction  
✅ Physical validation (0.01K < Tc < 300K)  
✅ Duplicate detection (44% flagged)  
✅ Quality tier classification  
✅ Zero missing critical data  

### Quality Flags
Each record includes 8+ flags:
- `quality_tier` - Strict/Standard/Inclusive
- `tc_validation` - Physical checks
- `has_oxygen_var` - O non-stoichiometry
- `is_duplicate_formula` - Multiple measurements
- `material_family` - Cuprate/iron-based/etc.
- `is_high_tc` - Above 77K (LN₂ temp)
- `n_elements` - Composition complexity
- `publication_year` - Discovery timeline

---

## Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Get started in 5 min
- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - Detailed findings
- **[analysis_template.ipynb](notebooks/analysis_template.ipynb)** - Jupyter examples

---

## Roadmap

- [x] **Phase 1:** Data cleaning & EDA ✅ Complete
- [ ] **Phase 2:** Feature engineering (elemental properties)
- [ ] **Phase 3:** Interactive dashboard (Plotly/Streamlit)
- [ ] **Phase 4:** Predictive modeling (optional)

---

## 📚 Citation

```bibtex
@misc{biswas2025superconductor,
  author = {Biswas, Ankita},
  title = {Superconductor Data Management Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/superconductor-data-pipeline}
}
```

**Data Source:** [NIMS Superconducting Materials Database](https://mdr.nims.go.jp/collections/4c428a0c-d209-4990-ad1f-656d05d1cfe2)

**Methods Inspired By:**
- Hamidieh (2018) *Comp. Mat. Sci.* 154:346-354
- Stanev et al. (2018) *npj Comp. Mat.* 4:29

---

## Author

**Ankita Biswas**  
PhD Candidate, Materials Informatics | University of Virginia  
Research: Thermoelectric materials, DFT, Machine Learning

---

## License 
NIMS data subject to their [terms of use](https://mdr.nims.go.jp/).

---

<div align="center">

**⭐ Star if useful! ⭐**

[Report Bug](https://github.com/yourusername/superconductor-data-pipeline/issues) · 
[Request Feature](https://github.com/yourusername/superconductor-data-pipeline/issues)

</div>
