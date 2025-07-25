**FraudNet**

A machine learning project that detects fraudulent transactions using behavioral patterns and transaction features. Built with Python and scikit-learn.

**Project Overview**

This fraud detection system analyzes digital transactions to identify potentially fraudulent activities by examining:

Transaction amounts

Transaction timing patterns

High-value transaction flags

Time-based behavioral indicators


**Key Results:**

54.11% Recall - Successfully catches 54.11% of fraud cases
318,131 Transactions processed
2 ML Algorithms compared (Random Forest, Rule-Based)

**Quick Start**
Prerequisites

Python 3.8+
pip package manager

**Installation**

1. **Clone the repository**

```git clone <your-repo-url>```

```cd machine_learning_finals```

2. **Create virtual environment**

```python3 -m venv fraud_detection_env```

```source fraud_detection_env/bin/activate``` 

On Windows: 

```source fraud_detection_env\Scripts\activate```

3. **Install dependencies**

```pip install -r requirements.txt```

4. **Running the Application**

```streamlit run streamlit.py```

**Model Performance**
### Best Model: Random Forest

| Metric | Score | Explanation |
|--------|-------|-------------|
| **Recall** | 54.11% | Catches 54.11% of actual fraud cases |
| **Precision** | 51.33% | 51.33% of flagged transactions are fraud |
| **F1-Score** | 52.68% | Balanced precision-recall measure |
| **Accuracy** | 97.49% | Overall prediction accuracy |

**How It Works**
1. **Data Processing**

Cleans transaction data
Handles missing values
Creates time-based features

2. **Feature Engineering**
python# Key features created:
- hour = step % 24                    # Hour of day (0-23)
- is_night = (hour >= 22) | (hour <= 6)  # Night time flag
- is_high_amount = amount > 50000     # High value flag

3. Model Training

Compares 2 algorithms: Random Forest, Rule-Based
Uses stratified sampling to handle class imbalance
Optimizes for fraud detection (high recall)

4. **Fraud Detection Logic**

FRAUD ALERT triggered when:

AI Model Probability ≥ 50% (Machine learning prediction)

No hard-coded rules for amount or timing

Pure model-based decision making

Risk Thresholds:

0-50% Risk: ✅ Transaction Approved (Process normally)

50-75% Risk: ⚠️ Manual Review Required (Contact bank)

76-100% Risk: 🚫 Transaction Blocked (Block and investigate)

**Requirements**
- txtpandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11