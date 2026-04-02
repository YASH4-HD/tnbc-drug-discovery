# TNBC Drug Discovery Pipeline

## Overview

A computational pipeline for discovering novel druggable targets in triple negative breast cancer (TNBC) using protein-protein interaction networks from [STRING-db](https://string-db.org).

**Key Features:**
- 🧬 Query STRING-db API for protein interactions
- 🎯 Filter for druggable surface receptors
- 📊 Rank targets by binding confidence + experimental evidence
- 📥 Export results as CSV/JSON
- 🎨 Interactive web interface (Streamlit)

## Scientific Background

### Triple Negative Breast Cancer (TNBC)
TNBC is characterized by lack of ER, PR, and HER2 expression, making it aggressive and difficult to treat. Current targets of interest include:
- **EGFR**: Overexpressed in many TNBCs
- **MMPs**: Matrix metalloproteinases (MMP-1, -7, -12) drive invasion and metastasis
- **DNMT**: DNA methyltransferases involved in epigenetic silencing

### Discovery Strategy
Instead of screening thousands of compounds, this pipeline:
1. Identifies proteins that directly interact with known TNBC drivers (EGFR, MMPs, DNMT)
2. Filters for druggable surface receptors (which can be targeted by antibodies, inhibitors, etc.)
3. Ranks by interaction strength + experimental validation
4. Produces a prioritized list of novel targets for functional studies

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tnbc-drug-discovery.git
cd tnbc-drug-discovery

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Local Deployment

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment

Deploy for free on Streamlit Cloud:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

## How It Works

### Step 1: Primary Target Query
The app queries STRING-db for all proteins that interact with:
- EGFR (Epidermal Growth Factor Receptor)
- MMP-1, MMP-7, MMP-12 (Matrix Metalloproteinases)
- DNMT1 (DNA Methyltransferase)

### Step 2: Druggable Receptor Filter
Interactions are filtered to include only known druggable surface receptors:
- **EGFR Family**: EGFR, ERBB2, ERBB3, ERBB4
- **Growth Factor Receptors**: MET, IGF1R, FGFR1/2/3, PDGFRA/B
- **Proteases**: MMP family members, ADAM proteases
- **Immune Targets**: CD40, TNF receptors, IL cytokine receptors
- **Developmental**: Notch, Wnt/Frizzled receptors
- **Cell Adhesion**: Integrin family, NCAM, L1CAM

### Step 3: Ranking
Targets are ranked by:
1. **Interaction Score** (50%): Confidence of protein-protein interaction
2. **Experimental Evidence** (30%): Direct experimental validation
3. **Interaction Frequency** (20%): Frequency of connections across primary targets

### Step 4: Export
Results exported as CSV/JSON with:
- Gene name and STRING ID
- Interaction scores from multiple sources (experiments, coexpression, database)
- Which primary target it interacts with

## Configuration Options

### Confidence Threshold
- Range: 0.4 - 1.0
- Default: 0.7
- Higher = more confident interactions (fewer false positives)

### Primary Targets
Select which TNBC drivers to query (default: all 5)

### Receptor Categories
Select which druggable protein classes to include

## Output Format

### CSV Output
```
Rank,Gene,Score,Experiments,Coexpression,Database,InteractsWith,StringID
1,ERBB3,0.987,0.876,0.654,0.543,EGFR,9606.ENSP00000263126
2,MET,0.954,0.823,0.712,0.601,EGFR,9606.ENSP00000245812
...
```

### JSON Output
```json
[
  {
    "Rank": 1,
    "Gene": "ERBB3",
    "Score": 0.987,
    "Experiments": 0.876,
    "Coexpression": 0.654,
    "Database": 0.543,
    "InteractsWith": "EGFR",
    "StringID": "9606.ENSP00000263126"
  }
]
```

## Data Sources

### STRING-db
- **Database**: STRING v12.0 (Human)
- **Interactions**: 20+ million protein associations
- **Evidence**: Experiments, databases, text mining, coexpression, homology
- **URL**: https://string-db.org
- **Citation**: Szklarczyk et al. (2023) Nucleic Acids Research

## Pipeline Assumptions

1. **Surface Localization**: Proteins are druggable only if they have extracellular/surface domains
2. **Confidence Score**: Threshold of 0.7 balances sensitivity and specificity
3. **Protein-Protein Interactions**: Direct physical interactions are prioritized
4. **Experimental Validation**: Presence of experimental evidence increases confidence

## Limitations & Future Work

### Current Limitations
- Focuses on direct protein-protein interactions (misses indirect effects)
- STRING-db data is static (updated periodically, not real-time)
- Does not account for tissue-specific expression
- No structural docking predictions (Phase 2 of project)

### Planned Features
- Phase 2: AutoDock Vina molecular docking of candidate compounds
- Phase 3: ADMET profiling (absorption, distribution, metabolism, toxicity)
- Phase 4: Integration with TCGA/CCLE expression data for TNBC-specific ranking
- Streamlit dashboard for visualization and interactive filtering

## Project Structure

```
tnbc-drug-discovery/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── string_db_targets.csv # Example output
└── docs/
    └── methodology.md        # Detailed methodology
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional target databases (DrugBank, ChEMBL)
- Integration with expression data (TCGA, CCLE)
- Molecular docking pipeline
- ADMET analysis
- Advanced visualization

## License

MIT License - see LICENSE file for details

## Contact & Attribution

**Author**: Yashwant (Independent Computational Researcher)
**Affiliation**: TNBC Drug Discovery Initiative
**GitHub**: [tnbc-drug-discovery](https://github.com/your-username/tnbc-drug-discovery)

## References

1. Szklarczyk, D., et al. (2023). "The STRING database in 2023: protein–protein association networks at your fingertips in both humans and mice." *Nucleic Acids Research*, 51(D1), D638-D646.

2. Bioinformatics STRING-db API: https://string-db.org/api/

3. TNBC Biology:
   - Reis-Filho, J. S., & Tutt, A. N. (2018). "Triple negative tumours: a critical review." *Journal of Clinical Oncology*.
   - Lehmann, B. D., et al. (2011). "Identification of human triple-negative breast cancer subtypes."

## Acknowledgments

- STRING-db team for comprehensive protein interaction data
- Streamlit for interactive web app framework
- Python scientific computing ecosystem (pandas, requests, numpy)

---

**Last Updated**: 2026-04-03  
**Status**: Active Development  
**Version**: 1.0.0
