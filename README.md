**Fraud Detection in Digital Transactions**

A machine learning project that detects fraudulent transactions using behavioral patterns and transaction features. Built with Python and scikit-learn.

**Project Overview**

This fraud detection system analyzes digital transactions to identify potentially fraudulent activities by examining:

Transaction amounts

Transaction timing patterns

High-value transaction flags

Time-based behavioral indicators


**Key Results:**

68.35% Recall - Successfully catches 68% of fraud cases
6.2M+ Transactions processed
3 ML Algorithms compared (Random Forest, Logistic Regression, Neural Network)

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

```python streamlit.py```

**Model Performance**
### Best Model: Random Forest

| Metric | Score | Explanation |
|--------|-------|-------------|
| **Recall** | 68.35% | Catches 68% of actual fraud cases |
| **Precision** | 1.41% | 1.4% of flagged transactions are fraud |
| **F1-Score** | 2.77% | Balanced precision-recall measure |
| **Accuracy** | 95.1% | Overall prediction accuracy |

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

Compares 3 algorithms: Random Forest, Logistic Regression, Neural Network
Uses stratified sampling to handle class imbalance
Optimizes for fraud detection (high recall)

4. **Fraud Detection Logic**
FRAUD ALERT triggered when:
- High transaction amount (₱50k+)
- Unusual timing (night hours)
- Combined risk patterns

**Requirements**
- txtpandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11