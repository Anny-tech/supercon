#  Normalized Database and Interactive Dashboard for Superconductors

> **Clean, explore, and visualize 26,000+ superconductor records from the NIMS database**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Data Quality](https://img.shields.io/badge/Data%20Quality-99.6%25-success.svg)]()
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

A reproducible data cleaning and visualization dashboard for promoting and pushing research on superconducturs a fascinating material system. 

---

## What This Does

Takes messy superconductor data → Cleans it rigorously → Provides analysis-ready datasets

**Input:** 26,357 raw records from NIMS database  
**Output:** 15,845 validated records + visualizations + interactive tools

---

## Quick Start

```bash
git clone https://github.com/Anny-tech/supercon.git
cd supercon
cd dashboard_app
docker-compose up --build
```

---

## Key Results

Data Statistics:
| Metric | Value |
|--------|-------|
| Final records | 15845 |
| Total columns | 164 |
| Composition features | 146 |
| Time Span | 1911-2025 |

Tc Statistics:
| Metric | Value |
|--------|-------|
|  Mean | 29.87 K |
|  Median | 14.40 K |
|  Std | 32.82 K |
|  Min | 0.01 K |
|  Max | 294.00 K |

Material Categories:
  Cuprate: 5910 (37.3%)
  Other: 5721 (36.1%)
  Iron-pnictide: 1124 (7.1%)
  Niobium-based: 994 (6.3%)
  Heavy-fermion: 560 (3.5%)
  Borocarbide: 308 (1.9%)
  Iron-chalcogenide: 267 (1.7%)
  Bismuthate: 253 (1.6%)
  Iron-based: 205 (1.3%)
  MgB2-type: 138 (0.9%)
  Niobium-nitride: 81 (0.5%)
  Organic: 72 (0.5%)
  Hydrogen-rich: 69 (0.4%)
  Elemental: 60 (0.4%)
  Niobium-selenide: 45 (0.3%)
  Cobaltate: 33 (0.2%)
  Ruthenate: 4 (0.0%)
  Mercury-cuprate: 1 (0.0%)

## Pipeline Features

### Data Cleaning
✅ Formula parsing & element extraction  
✅ Physical validation (0.01K < Tc < 300K)  
✅ Duplicate detection (44% flagged)  
✅ Quality tier classification  
✅ Zero missing critical data  

---

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

## Roadmap

- [x] **Phase 1:** Data cleaning & EDA ✅ Complete
- [x] **Phase 2:** Feature engineering (elemental properties)
- [x] **Phase 3:** Interactive dashboard (Plotly/Streamlit)

---

## Citation

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
