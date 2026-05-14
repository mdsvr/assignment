# MDS Static Model for Microarchitectural Data Sampling Analysis

This project implements a comprehensive static model for MDS (Microarchitectural Data Sampling) attack detection based on the research document: **"Creating Static Data for MDS Analysis" (May 2026)**.

## Overview

The static model provides a complete framework for:
- Generating synthetic datasets for MDS detection research
- Engineering features from Hardware Performance Counters (HPCs)
- Validating dataset quality
- Training ML models for MDS attack detection

## Project Structure

```
.
├── mds_static_model.py              # Core data structures and HPC collection framework
├── feature_engineering.py          # Feature extraction (raw, derived, MDS-specific, temporal)
├── synthetic_data_generator.py      # Synthetic data generation with workload profiles
├── dataset_validator.py            # Dataset validation and analysis tools
├── mds_detector.py                 # ML-based detection models
├── generate_mds_dataset.py          # Standalone script to generate datasets
├── mds_dataset.csv                 # Generated synthetic dataset (example)
└── README.md                       # This file
```

## Generated Dataset

The dataset (`mds_dataset.csv`) follows the schema specified in PDF Section 6.1:

### Schema
- `timestamp`: Unix timestamp of sample
- `sample_id`: Unique sample identifier
- `run_id`: Experiment run identifier (for proper train/test splitting)
- `llc_load_misses`: LLC load misses count
- `l1d_load_misses`: L1 data cache load misses
- `branch_misses`: Branch misprediction count
- `branch_instr`: Total branch instructions
- `instructions`: Instructions retired
- `cache_references`: Cache reference count
- `page_faults`: Page fault count
- `context_switches`: Context switch count
- `cpu_cycles`: CPU cycles elapsed
- `cache_miss_ratio`: Derived feature (misses/references)
- `ipc`: Derived feature (instructions/cycles)
- `branch_miss_rate`: Derived feature (misses/branches)
- `label`: Class label (benign_cpu, benign_mem, benign_io, benign_mixed, msbds, mfbds, mlpds, mdsum)
- `attack_variant`: MDS variant (if attack)

### Class Distribution
The generated dataset includes:
- **Benign Workloads** (54.5%):
  - `benign_cpu`: CPU-intensive normal workload (22.7%)
  - `benign_mem`: Memory-intensive normal workload (13.6%)
  - `benign_io`: I/O-intensive normal workload (9.1%)
  - `benign_mixed`: Mixed normal workload (9.1%)

- **MDS Attack Variants** (45.5%):
  - `msbds`: Store Buffer Data Sampling attack (13.6%)
  - `mfbds`: Fill Buffer Data Sampling attack (13.6%)
  - `mlpds`: Load Port Data Sampling attack (9.1%)
  - `mdsum`: Uncacheable Memory Sampling (9.1%)

## Usage

### 1. Generate a New Dataset

```bash
# Generate dataset with default settings (10,000 samples)
python generate_mds_dataset.py

# Generate custom dataset
python generate_mds_dataset.py --n_samples 20000 --output my_dataset.csv --samples_per_run 100
```

**Options:**
- `--n_samples`: Total number of samples (default: 10000)
- `--output`: Output CSV file path (default: mds_synthetic_dataset.csv)
- `--samples_per_run`: Samples per experimental run (default: 100)
- `--sampling_interval`: Sampling interval in milliseconds (default: 100ms)
- `--random_seed`: Random seed for reproducibility (default: 42)

### 2. Load and Analyze the Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('mds_dataset.csv')

# View basic statistics
print(df.describe())
print(df['label'].value_counts())

# Separate attack and benign samples
attack_labels = ['msbds', 'mfbds', 'mlpds', 'mdsum']
attack_df = df[df['label'].isin(attack_labels)]
benign_df = df[~df['label'].isin(attack_labels)]
```

### 3. Feature Engineering

```python
from feature_engineering import MDSFeatureEngineer

# Initialize feature engineer
engineer = MDSFeatureEngineer(window_size=10)

# Extract all features
df_with_features = engineer.extract_all_features(df)

# Get feature importance scores
importance = engineer.get_feature_importance_scores(df_with_features)
print("Top features:", list(importance.keys())[:10])
```

### 4. Dataset Validation

```python
from dataset_validator import DatasetValidator

# Initialize validator
validator = DatasetValidator(df)

# Run statistical validation
validator.statistical_validation()

# Run separability analysis
validator.separability_analysis()

# Generate validation report
print(validator.generate_validation_report())
```

### 5. Train MDS Detection Model

```python
from mds_detector import MDSDetector

# Initialize detector
detector = MDSDetector(model_type='random_forest')

# Split data by run_id (prevents data leakage)
X_train, X_val, X_test, y_train, y_val, y_test = detector.split_by_run(df)

# Train the model
detector.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = detector.evaluate(X_test, y_test)
detector.print_evaluation_report(metrics)

# Get feature importance
importance = detector.get_feature_importance()
```

## Key Features

### 1. HPC Data Collection
- Implements Hardware Performance Counter event definitions from PDF Section 2.1
- Supports 8 key HPC events for MDS detection
- Follows 100ms sampling interval recommendation from PDF Section 3.3

### 2. Feature Engineering
- **Raw Counter Features**: Absolute values, rates, basic ratios (PDF 4.1)
- **Derived Features**: Statistical features over sliding windows (PDF 4.2)
- **MDS-Specific Features**: Tailored for MDS detection (PDF 4.3)
- **Temporal Features**: Time-series analysis (PDF 4.4)

### 3. Synthetic Data Generation
- Workload profiles based on PDF Sections 5.1 and 5.2
- Poisson distribution for realistic HPC simulation
- Run-based splitting for proper ML evaluation
- Controlled environment simulation

### 4. Dataset Validation
- Statistical validation (class balance, feature variance, outliers)
- Separability analysis (statistical tests, mutual information)
- Cross-validation considerations (run-level splitting)

### 5. ML Detection Models
- Multiple model types: Random Forest, Gradient Boosting, Logistic Regression, SVM, Neural Network
- Run-level splitting to prevent data leakage
- Comprehensive evaluation metrics
- Feature importance analysis

## Methodology

This implementation follows the methodology described in the research document:

1. **Data Sources**: Uses Hardware Performance Counters (HPCs) as primary data source (Section 2)
2. **Collection**: Implements perf-tool methodology with 100ms intervals (Section 3)
3. **Feature Engineering**: Four-tier feature extraction approach (Section 4)
4. **Data Generation**: Synthetic data based on workload profiles (Section 5)
5. **Schema Design**: CSV format following PDF specification (Section 6)
6. **Labeling Strategy**: Binary and multi-class labeling (Section 8)
7. **Validation**: Statistical and separability validation (Section 9)

## Requirements

```
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
PyPDF2
```

Install dependencies:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn PyPDF2
```

## Example Workflow

```python
# 1. Generate dataset
from generate_mds_dataset import MDSDatasetGenerator

generator = MDSDatasetGenerator(sampling_interval_ms=100)
df = generator.generate_default_dataset(n_samples=10000)
generator.save_dataset(df, 'my_mds_dataset.csv')

# 2. Validate dataset
from dataset_validator import DatasetValidator

validator = DatasetValidator(df)
validator.statistical_validation()
validator.separability_analysis()
print(validator.generate_validation_report())

# 3. Engineer features
from feature_engineering import MDSFeatureEngineer

engineer = MDSFeatureEngineer(window_size=10)
df_features = engineer.extract_all_features(df)

# 4. Train detector
from mds_detector import MDSDetector

detector = MDSDetector(model_type='random_forest')
X_train, X_val, X_test, y_train, y_val, y_test = detector.split_by_run(df_features)
detector.train(X_train, y_train, X_val, y_val)
metrics = detector.evaluate(X_test, y_test)
detector.print_evaluation_report(metrics)
```

## Notes

- The generated dataset uses run-based splitting to prevent data leakage, as recommended in PDF Section 6.3
- Feature engineering can be computationally intensive for large datasets
- The ML detector supports multiple model types - Random Forest is recommended for most use cases
- Sampling interval of 100ms provides optimal tradeoff between resolution and manageability (PDF Section 3.3)

## References

This implementation is based on the research document: **"Creating Static Data for MDS Analysis" (May 2026)**

Key references from the document:
- HARPY Dataset: "Hardware Attack detectoR via Performance counters analYsis"
- MAD-EN: "Microarchitectural Attack Detection through Energy Consumption"
- MADFAM: "MicroArchitectural Data Framework and Methodology"
