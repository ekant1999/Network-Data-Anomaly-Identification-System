# Network Data Anomaly Identification System

A comprehensive machine learning system for identifying anomalies in network data, specifically focused on detecting DoS (Denial of Service) and DDoS (Distributed Denial of Service) attacks in network traffic.

## Overview

This project implements a complete end-to-end Network Intrusion Detection System (NIDS) that analyzes network traffic data to identify malicious patterns and anomalies. The system processes network flow data from the CICIDS 2017 dataset through a 4-step pipeline: Data Preprocessing, Outlier Detection & Removal, Feature Engineering, and Model Training & Evaluation. The best performing model achieves **99.76% accuracy** with **99.41% F1-Score** and **99.98% ROC-AUC** for DoS/DDoS attack detection.

## Features

- **Complete ML Pipeline**: 4-step end-to-end pipeline from data loading to model evaluation
- **Data Preprocessing**: Automated loading, normalization, missing value handling, and duplicate removal
- **Outlier Detection**: IQR-based outlier detection with domain constraint validation
- **Feature Engineering**: Statistical feature selection (ANOVA F-test) with top 50 features selected
- **Model Training**: Multiple ML models (Logistic Regression, Random Forest, Gradient Boosting) with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) with visualizations
- **Best Model Performance**: Random Forest achieves 99.76% accuracy, 99.41% F1-Score, and 99.98% ROC-AUC

## Project Structure

```
Network-Data-Anomaly-Identification-System/
├── src/
│   ├── load_data.py          # Data loading from ZIP archives
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature extraction and engineering
│   ├── models.py              # Machine learning model definitions
│   ├── evaluation.py          # Model evaluation metrics
│   ├── utils.py               # Utility functions
│   └── __init__.py
├── notebooks/
│   ├── NIDS_255.ipynb                              # Main data exploration notebook
│   └── NIDS_255__Model_Training_&_Evaluation.ipynb # Model training and evaluation
├── config/                     # Configuration files
├── docs/                       # Documentation
├── results/                     # Model outputs and results
├── train.py                    # Training script
└── README.md                   # This file
```

## Requirements

### Python Packages
- pandas
- numpy
- scikit-learn
- jupyter (for notebooks)
- Additional packages may be required (see requirements.txt if available)

### Data
The system expects network traffic data in CSV format from the **CICIDS 2017 dataset**. The data should be organized in a ZIP archive with the following structure:
- `TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS attacks)
- `TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv` (Multiple DoS variants)
- `TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv` (Mixed traffic)

**Dataset Statistics:**
- Initial dataset: 1,364,357 rows × 86 columns
- After preprocessing: 670,699 rows × 86 columns
- Attack distribution: 20.78% DoS/DDoS, 79.22% BENIGN

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ekant1999/Network-Data-Anomaly-Identification-System.git
cd Network-Data-Anomaly-Identification-System
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn jupyter matplotlib seaborn pyarrow
```

## Usage

### Complete Pipeline

The main workflow is implemented in the Jupyter notebook `NIDS_255__Model_Training_&_Evaluation.ipynb`. This notebook provides a complete 4-step pipeline:

#### Step 1: Data Preprocessing
- Load and combine multiple CSV files from ZIP archive
- Normalize column names (handle BOM, whitespace, special characters)
- Handle missing values (fill with median for numeric columns)
- Remove duplicate rows (9,880 duplicates removed)
- **Output**: 1,354,477 rows × 86 columns

#### Step 2: Outlier Detection & Removal
- Fix negative values in domain-constrained columns (52 rows removed)
- Remove zero-duration flows (1,570 flows removed)
- Apply IQR-based outlier detection (3×IQR threshold)
- Remove extreme outliers from critical columns (683,778 rows removed)
- Domain constraint validation
- **Output**: 670,699 rows × 86 columns

#### Step 3: Feature Engineering
- Create binary target variable (BENIGN vs DoS/DDoS)
- Select top 50 features using ANOVA F-test (f_classif)
- Perform train-test split (70/30, stratified)
- Standardize features using StandardScaler
- **Output**: Training set (469,489 samples), Test set (201,210 samples)

#### Step 4: Model Training & Evaluation
- Train multiple models: Logistic Regression, Random Forest, Gradient Boosting
- Evaluate with comprehensive metrics
- Generate visualizations (ROC curves, confusion matrices, feature importance)
- Save best model and results

### Running the Notebook

1. Open the training notebook:
```bash
jupyter notebook notebooks/NIDS_255__Model_Training_&_Evaluation.ipynb
```

2. Update the data path in the configuration cell:
```python
ZIP_PATH = "/path/to/your/GeneratedLabelledFlows.zip"
OUTPUT_DIR = "/path/to/output/directory"
```

3. Run all cells sequentially to execute the complete pipeline

## Key Components

### Data Preprocessing
- **Column Normalization**: Removes BOM, whitespace, and special characters
- **Missing Value Handling**: Fills numeric columns with median (0.09% missing)
- **Duplicate Removal**: Removes feature-identical rows (0.72% duplicates)
- **Data Type Conversion**: Ensures proper data types for all columns

### Outlier Detection & Removal
- **Domain Violations**: Detects and removes negative values in constrained columns
- **Zero-Duration Flows**: Removes flows with zero duration (causes division errors)
- **IQR-Based Outlier Detection**: Uses 3×IQR threshold for less aggressive removal
- **Domain Constraints**: Validates port ranges (0-65535), protocol ranges (0-255)

### Feature Engineering
- **Binary Target Creation**: Maps labels to BENIGN (0) vs DoS/DDoS (1)
- **Feature Selection**: Selects top 50 features using ANOVA F-test
- **Feature Scaling**: Standardizes features using StandardScaler (mean=0, std=1)
- **Train-Test Split**: 70/30 stratified split to maintain class distribution

### Models
The system trains and compares multiple machine learning models:
- **Logistic Regression**: Baseline linear model (98.30% accuracy)
- **Random Forest**: Best performing model (99.76% accuracy, 99.41% F1-Score)
- **Gradient Boosting**: Strong ensemble model (99.54% accuracy)

### Evaluation Metrics
- **Accuracy**: Overall correctness (99.76%)
- **Precision**: True positives / (True positives + False positives) (99.28%)
- **Recall**: True positives / (True positives + False negatives) (99.54%)
- **F1-Score**: Harmonic mean of precision and recall (99.41%)
- **ROC-AUC**: Area under ROC curve (99.98%)
- **Confusion Matrix**: Detailed breakdown of predictions

## Dataset

This project uses the **CICIDS 2017 (CIC Intrusion Detection System 2017)** dataset, which contains labeled network traffic flows including:

### Attack Types
- **BENIGN**: Normal network traffic (79.22% of dataset)
- **DoS Attacks**: 
  - DoS Hulk (5.70%)
  - DDoS (13.42%)
  - DoS GoldenEye (1.22%)
  - DoS slowloris (0.30%)
  - DoS SlowHTTPTest (0.15%)
- **Other Attacks**: FTP-Patator, SSH-Patator (minority classes)

### Dataset Characteristics
- **Source Files**: 3 CSV files from different time periods
- **Initial Size**: 1,364,357 rows × 86 columns
- **After Preprocessing**: 670,699 rows × 86 columns
- **Features**: Network flow characteristics (packet counts, durations, protocols, flags, etc.)
- **Target**: Binary classification (BENIGN vs DoS/DDoS)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

**ekant1999**

## Acknowledgments

- CICIDS 2017 dataset providers (Canadian Institute for Cybersecurity)
- Contributors and maintainers of open-source ML libraries (scikit-learn, pandas, numpy)

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|-----------|---------|
| **Random Forest** | **0.9976** | **0.9928** | **0.9954** | **0.9941** | **0.9998** |
| Gradient Boosting | 0.9954 | 0.9843 | 0.9939 | 0.9890 | 0.9999 |
| Logistic Regression | 0.9830 | 0.9508 | 0.9686 | 0.9596 | 0.9969 |

### Key Findings
- **Best Model**: Random Forest Classifier
- **False Positive Rate**: 0.19% (300 out of 159,394 benign samples)
- **False Negative Rate**: 0.46% (191 out of 41,816 attack samples)
- **Top Features**: Packet Length Std, Packet Length Mean, Average Packet Size, Bwd Packet Length Std

## Future Enhancements

- [ ] Real-time network traffic analysis
- [ ] Support for additional attack types (FTP-Patator, SSH-Patator, etc.)
- [ ] Deep learning model implementations (LSTM, CNN)
- [ ] Hyperparameter tuning with GridSearch/RandomSearch
- [ ] Cross-validation for robust evaluation
- [ ] API for deployment (Flask/FastAPI)
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

## Contact

For questions or issues, please open an issue on the GitHub repository.

