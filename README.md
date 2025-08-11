# 🤖 Machine Learning Data Preprocessing Pipeline

A comprehensive data preprocessing project demonstrating essential ML engineering skills including data cleaning, feature engineering, encoding techniques, and data preparation for machine learning models.

## 📋 Project Overview

This project implements a complete data preprocessing pipeline for a demographic and income dataset, showcasing industry-standard practices for preparing raw data for machine learning applications. The pipeline handles real-world data challenges including missing values, outliers, categorical encoding, feature scaling, and proper train-test splitting.

## 🎯 Key Features

- **Data Quality Management**: Automated handling of missing values and outlier detection
- **Duplicate Detection**: Identification and removal of duplicate records
- **Advanced Encoding Techniques**: Implementation of both Label Encoding and One-Hot Encoding
- **Feature Scaling**: Standardization and normalization of numerical features
- **ML-Ready Data Preparation**: Proper train-test splitting for model development

## 📊 Dataset Information

**Source**: Custom demographic dataset (`sample_data.csv`)
- **Records**: 50 entries (47 after cleaning)
- **Features**: 5 columns (Age, Gender, City, Education, Income)
- **Target Variable**: Income (continuous)
- **Data Challenges**: Missing values, outliers (Age > 100), duplicate records

### Dataset Schema
```
Age         : float64  - Age in years
Gender      : object   - Male/Female
City        : object   - Los Angeles/New York/Houston/Chicago
Education   : object   - High School/Bachelor/Master/PhD
Income      : float64  - Annual income in USD
```

## 🛠 Technical Implementation

### Dependencies
```toml
# pyproject.toml
[project]
dependencies = [
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "numpy>=1.21.0"
]
```

### Core Components

#### 1. Data Cleaning Module
- **Missing Value Imputation**:
  - Categorical: Mode imputation for Gender
  - Numerical: Median imputation for Income
- **Outlier Treatment**: Age values >100 replaced with median
- **Duplicate Removal**: 3 duplicate records identified and removed

#### 2. Feature Engineering Pipeline
- **Label Encoding**: Binary encoding for Gender (Male=1, Female=0)
- **One-Hot Encoding**: Categorical expansion for Education and City features
- **Feature Scaling**: MinMaxScaler applied to numerical features (Age, Income)

#### 3. Data Preparation
- **Train-Test Split**: 80/20 stratified split with random_state=42
- **Final Feature Set**: 9 engineered features ready for ML models

## 🚀 Usage

```bash
# Clone the repository
git clone [your-repo-url]

# Navigate to project directory
cd ml-data-preprocessing

# Install dependencies using uv
uv sync

# Run the preprocessing pipeline
uv run python data_preprocessing.py
```

## 📈 Results & Output

### Data Transformation Summary
- **Original Dataset**: 50 rows × 5 columns
- **Cleaned Dataset**: 47 rows × 5 columns (3 duplicates removed)
- **Final Feature Matrix**: 47 rows × 9 columns (after encoding)
- **Training Set**: 38 samples
- **Test Set**: 9 samples

### Generated Features
```
Original: Age, Gender, City, Education, Income
Processed: Age, Gender_Encoded, Education_Bachelor, Education_High School, 
          Education_Master, Education_PhD, City_Chicago, City_Houston, 
          City_Los Angeles, City_New York, Income (scaled)
```

## 🔧 Technical Skills Demonstrated

- **Data Wrangling**: Pandas proficiency for data manipulation
- **Statistical Analysis**: Understanding of data distributions and outlier detection
- **Feature Engineering**: Advanced categorical encoding techniques
- **ML Best Practices**: Proper data splitting and preprocessing pipelines
- **Code Quality**: Clean, documented, and modular Python implementation

## 📝 Key Learnings

1. **Data Quality**: Real-world datasets require extensive cleaning and validation
2. **Encoding Strategy**: Choice between Label vs One-Hot encoding based on feature characteristics
3. **Scaling Importance**: Numerical feature normalization for algorithm performance
4. **Pipeline Design**: Modular approach enables reproducible preprocessing workflows

## 🔮 Future Enhancements

- [ ] Implement cross-validation for robust model evaluation
- [ ] Add automated feature selection techniques
- [ ] Integrate with MLflow for experiment tracking
- [ ] Develop custom preprocessing transformers
- [ ] Add data validation and schema enforcement

## 🎓 Skills Portfolio

This project demonstrates proficiency in:
- **Machine Learning Engineering**: End-to-end ML pipeline development
- **Data Science**: Statistical analysis and feature engineering
- **Python Programming**: Advanced pandas, scikit-learn implementation
- **Best Practices**: Code organization, documentation, reproducibility



