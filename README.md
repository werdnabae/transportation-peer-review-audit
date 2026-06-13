# Code and Data: Peer Review Governance in Transportation Research

This folder contains the dataset and analysis code for the paper:

**"Peer Review Governance in Transportation Research: A Systematic Audit of Editorial Practices Across 108 Major Transportation Journals"**

Andrew J. Bae. *Transportation Research Interdisciplinary Perspectives* **38** (2026) 102058.

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.trip.2026.102058-blue)](https://doi.org/10.1016/j.trip.2026.102058)

**Paper:** <https://www.sciencedirect.com/science/article/pii/S259019822600223X>

## Contents

| File | Description |
|------|-------------|
| `transportation_journal_peer_review_dataset.xlsx` | Complete verified dataset of 108 transportation journals with peer review model classifications |
| `peer_review_analysis.py` | Python script that reproduces all tables, chi-square tests, and logistic regression results reported in the paper |

## Dataset

`transportation_journal_peer_review_dataset.xlsx` contains one row per journal (N = 108) with the following fields:

| Column | Description |
|--------|-------------|
| `Journal_Name` | Full journal title |
| `Publisher` | Publisher or publishing organisation |
| `Peer_Review_Model` | Coded review model: `Single-blind`, `Double-blind`, or `Unknown` |
| `Impact_Tier` | Impact tier based on OpenAlex 2-year mean citedness: `High` (>= 4.0), `Medium` (1.5-3.99), `Low` (< 1.5 or not indexed) |
| `OA_Status` | Open access status: `Subscription`, `FullOA` (Gold OA), or `Diamond` (no APCs) |
| `Year_Founded` | Original founding year of the journal |

### Verification protocol

Each journal's peer review model was verified by direct browser inspection of the journal's own author guidelines or editorial policy page during March 2026. The exact wording from each journal's policy page was recorded. No publisher-level defaults were assumed. Journals coded as `Unknown` had no explicit peer review model statement on their own guidelines page at the time of verification.

### Impact tier classification

Impact tiers are based on 2-year mean citedness retrieved from the OpenAlex API. Thresholds:
- **High**: 2-year mean citedness >= 4.0
- **Medium**: 1.5 to 3.99
- **Low**: < 1.5, not indexed, or insufficient data

## Analysis Script

`peer_review_analysis.py` reproduces all quantitative results reported in the paper:

- **Table 1**: Overall distribution of peer review models
- **Table 2**: Peer review model by publisher
- **Table 3**: Peer review model by impact tier and OA status
- **Table 4**: Peer review model by founding decade
- **Chi-square tests**: Publisher group vs. review model, impact tier vs. review model, OA status vs. review model
- **Logistic regression**: Predictors of single-blind review adoption
- **Key summary statistics**: Per-publisher breakdowns and high-impact tier analysis

### Requirements

```
python >= 3.8
pandas
numpy
scipy
openpyxl
scikit-learn
```

### Usage

```bash
pip install pandas numpy scipy openpyxl scikit-learn
python peer_review_analysis.py
```

The script reads the dataset from the same directory and prints all tables and test results to stdout.

## Citation

If you use this dataset or code, please cite the paper:

> Bae, A. J. (2026). Peer review governance in transportation research: a systematic audit of editorial practices across 108 major transportation journals. *Transportation Research Interdisciplinary Perspectives*, 38, 102058. https://doi.org/10.1016/j.trip.2026.102058

BibTeX:

```bibtex
@article{Bae2026PeerReview,
  title     = {Peer review governance in transportation research: a systematic audit of editorial practices across 108 major transportation journals},
  author    = {Bae, Andrew J.},
  journal   = {Transportation Research Interdisciplinary Perspectives},
  volume    = {38},
  pages     = {102058},
  year      = {2026},
  doi       = {10.1016/j.trip.2026.102058},
  publisher = {Elsevier},
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided, from which GitHub renders a "Cite this repository" button.

## License

This dataset is provided for research purposes. Please cite the paper if you use it in your work.
