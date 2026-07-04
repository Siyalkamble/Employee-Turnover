# Employee Turnover Prediction
### Supervised ML Assignment 2 — TalentCore Pvt. Ltd.

---

## Problem Statement

A multinational company **TalentCore Pvt. Ltd.** is experiencing rising employee resignations, increasing recruitment costs, project delays, and loss of skilled talent.

**Goal:** Build an intelligent ML system to predict whether an employee is likely to leave the company based on job satisfaction, salary, age, work-life balance, training hours, bonuses, and other work-related factors.

**As an AI/ML Engineer, the task is to:**
1. Build a baseline Logistic Regression model
2. Improve it using Regularization (L1 & L2)
3. Compare performances and recommend the best model

---

## How to Run

### Prerequisites
- Python 3.8+
- Git

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Siyalkamble/Employee-Turnover.git
   cd Employee-Turnover
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Logistic_Regression.ipynb
   ```

5. **Run the analysis:**
   - Open the notebook in your browser
   - Execute cells sequentially (Shift+Enter) or run all (Cell → Run All)
   - The notebook will load `employee_turnover.csv`, train three Logistic Regression models (Baseline, L1, L2), and display performance comparisons

### Expected Output
- Train/Test split: 1080 / 270 samples
- Model performance metrics, confusion matrices, and ROC-AUC scores
- Feature coefficients and importance rankings
- Comparison table showing L2 (Ridge) achieves the best accuracy (88.5%) and ROC-AUC (0.9582)


## Dataset Overview

- **Rows:** 1350 | **Columns:** 16 (15 features + 1 target)
- **Target:** `Employee_Turnover` (1 = Left, 0 = Stayed)
- **Pre-processed:** MinMax scaled (0–1), categoricals pre-encoded

| Feature | Description |
|---------|-------------|
| Job_Satisfaction | Level of satisfaction with the job |
| Performance_Rating | Employee performance score |
| Years_At_Company | Number of years worked |
| Work_Life_Balance | Balance between work and personal life |
| Distance_From_Home | Distance of home from workplace |
| Monthly_Income | Monthly salary |
| Education_Level | Education qualification level |
| Age | Age of employee |
| Num_Companies_Worked | Number of companies worked previously |
| Employee_Role | Encoded job role |
| Annual_Bonus | Bonus received annually |
| Training_Hours | Training hours attended |
| Department | Encoded department |
| Annual_Bonus_Squared | Engineered feature (bonus²) — **removed** |
| Annual_Bonus_Training_Hours_Interaction | Interaction feature |
| **Employee_Turnover** | **Target: 1 = Left, 0 = Stayed** |

---

## Phase 1 — Data Loading & Inspection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              ConfusionMatrixDisplay)

df = pd.read_csv('employee_turnover.csv')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
```

### Phase 1 Findings
- **Shape:** 1350 rows × 16 columns
- **Null values:** 0 — dataset is completely clean
- **Data types:** 15 float64 + 1 int64 (target)
- **Scaling:** All features already MinMax scaled (0–1)
- **Encoding:** Categorical columns (Employee_Role, Department) pre-encoded to float
- **Class balance:** Target mean = 0.497 → approximately 50/50 split → no class imbalance

---

## Phase 2 — Exploratory Data Analysis (EDA)

```python
# Target distribution
fig, ax = plt.subplots(figsize=(5, 4))
df['Employee_Turnover'].value_counts().plot(kind='bar', ax=ax,
    color=['#1D9E75', '#E24B4A'])
ax.set_title('Target Distribution')
ax.set_xticklabels(['Stayed (0)', 'Left (1)'], rotation=0)
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()

# KDE plots for key features
features_to_check = ['Job_Satisfaction', 'Years_At_Company', 'Num_Companies_Worked']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, feature in enumerate(features_to_check):
    df[df['Employee_Turnover'] == 0][feature].plot(
        kind='kde', ax=axes[i], label='Stayed', color='#1D9E75')
    df[df['Employee_Turnover'] == 1][feature].plot(
        kind='kde', ax=axes[i], label='Left', color='#E24B4A')
    axes[i].set_title(f'{feature} by Turnover')
    axes[i].legend()
plt.tight_layout()
plt.show()

# Box plots for all features
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
features = [col for col in df.columns if col != 'Employee_Turnover']
for i, feature in enumerate(features):
    df.boxplot(column=feature, by='Employee_Turnover', ax=axes[i])
    axes[i].set_title(feature)
    axes[i].set_xlabel('0=Stayed, 1=Left')
plt.suptitle('Feature Distribution by Turnover')
plt.tight_layout()
plt.show()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

### EDA Summary

#### Strong Predictors
`Job_Satisfaction`, `Performance_Rating`, `Years_At_Company`, `Work_Life_Balance`, and `Distance_From_Home` showed the strongest signal for predicting employee turnover. In the box plots, these features had noticeably different medians between employees who left vs employees who stayed — meaning the model can use these features to separate the two groups.

#### Surprising Finding
`Num_Companies_Worked` was expected to be a strong predictor based on the job-hopper hypothesis — someone who has switched multiple companies before is more likely to switch again. However, the KDE plot showed significant overlap between leavers and stayers for this feature, suggesting it is a weaker predictor than expected in this dataset. This is a good reminder that domain intuition is a starting point, not a guarantee.

#### Feature Removed
`Annual_Bonus_Squared` was removed due to multicollinearity (Pearson r = 0.97 with `Annual_Bonus`). Since it is directly derived from `Annual_Bonus`, keeping both would give the model redundant information and produce unstable coefficients in Logistic Regression.

```python
# Remove multicollinear feature
df = df.drop(columns=['Annual_Bonus_Squared'])
print(df.shape)  # (1350, 15)
```

---

## Phase 3 — Feature Engineering & Train/Test Split

```python
# Define features and target
X = df.drop(columns=['Employee_Turnover'])
y = df['Employee_Turnover']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTrain size: {X_train.shape}")
print(f"Test size:  {X_test.shape}")
print(f"\nTrain turnover rate: {y_train.mean():.3f}")
print(f"Test turnover rate:  {y_test.mean():.3f}")
```

### Phase 3 Summary
- **Split:** 80% train (1080 samples), 20% test (270 samples)
- **Stratified split** used to preserve class balance across both sets
- **Train turnover rate:** 0.498 | **Test turnover rate:** 0.496
- **No additional scaling applied** — data is already MinMax scaled (0–1)
- **Features:** 14 | **Target:** Employee_Turnover

---

## Phase 4 — Modeling

```python
# Helper function to evaluate any model
def evaluate_model(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred,
          target_names=['Stayed', 'Left']))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

    return {
        'Model'    : name,
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1'       : f1_score(y_test, y_pred),
        'ROC-AUC'  : roc_auc_score(y_test, y_proba)
    }

# Model 1: Baseline (no regularization)
baseline = LogisticRegression(
    C=np.inf,
    max_iter=1000,
    random_state=42
)
baseline.fit(X_train, y_train)
r1 = evaluate_model("Baseline (No Regularization)", baseline, X_test, y_test)

# Model 2: L1 Regularization (Lasso)
l1_model = LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
l1_model.fit(X_train, y_train)
r2 = evaluate_model("L1 Regularization (Lasso)", l1_model, X_test, y_test)

# Model 3: L2 Regularization (Ridge)
l2_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
l2_model.fit(X_train, y_train)
r3 = evaluate_model("L2 Regularization (Ridge)", l2_model, X_test, y_test)

# Comparison table
results = pd.DataFrame([r1, r2, r3])
results = results.set_index('Model')
print(results.round(4))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, model, name in zip(axes,
                            [baseline, l1_model, l2_model],
                            ['Baseline', 'L1', 'L2']):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['Stayed', 'Left'],
        ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# L1 Feature Coefficients
coef = pd.Series(
    l1_model.coef_[0],
    index=X_train.columns
).sort_values(key=abs, ascending=False)

print("\nL1 Coefficients (sorted by importance):")
print(coef)
print(f"\nFeatures zeroed out by L1: {(coef == 0).sum()}")

fig, ax = plt.subplots(figsize=(10, 5))
coef.plot(kind='bar', ax=ax,
          color=['#E24B4A' if v < 0 else '#1D9E75' for v in coef])
ax.set_title('L1 Feature Coefficients')
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# L1 vs L2 coefficient comparison
coef_comparison = pd.DataFrame({
    'L1': l1_model.coef_[0],
    'L2': l2_model.coef_[0]
}, index=X_train.columns).round(4)

print(coef_comparison)
print(f"\nFeatures zeroed by L1: {(coef_comparison['L1'] == 0).sum()}")
print(f"Features zeroed by L2: {(coef_comparison['L2'] == 0).sum()}")
```

---

## Phase 5 — Model Comparison & Recommendation

### Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Baseline (No Regularization) | 0.8778 | 0.8915 | 0.8582 | 0.8745 | 0.9575 |
| L1 Regularization (Lasso) | 0.8815 | 0.8923 | 0.8657 | 0.8788 | 0.9584 |
| L2 Regularization (Ridge) | 0.8852 | 0.8992 | 0.8657 | 0.8821 | 0.9582 |

### ✅ Recommended Model: L2 Regularization (Ridge)

L2 achieves the highest Accuracy (88.5%) and Precision (89.9%) while matching L1 on Recall (86.6%). ROC-AUC of 0.9582 confirms excellent class separation between leavers and stayers.

### Why not Baseline?
Baseline has no regularization — slightly lower performance suggests minor overfitting on training data. Both L1 and L2 correct this through weight penalization.

### Why L2 over L1?
Both models perform nearly identically on metrics. However, L2 is preferred here because:
- The dataset has no truly useless features — L1 only zeroed out 1 feature (the interaction term), confirming most features carry signal
- L2 distributes weight across correlated features more stably, which suits this dataset structure
- L1 is preferred when aggressive feature selection is needed — not the case here

### Key Metric: Recall
For employee turnover prediction, **Recall is the most critical metric**. A False Negative means an employee who will leave is predicted to stay — HR takes no action, the employee resigns, and the company incurs rehiring costs of 6–9 months salary per person. Our L2 model catches **86.6% of actual leavers** (117 out of 134 in the test set).

### Feature Insights from L1 Coefficients

**Top predictors of turnover (positive coefficients):**
- `Performance_Rating` (5.53) — high performers get poached by competitors
- `Job_Satisfaction` (5.52) — strong satisfaction signal
- `Work_Life_Balance` (5.31) — balance directly impacts retention decisions
- `Distance_From_Home` (5.10) — long commutes increase attrition risk
- `Years_At_Company` (4.76) — longer tenure may indicate stagnation

**Predictors of staying (negative coefficients):**
- `Education_Level` (-0.85) — higher education correlates with staying
- `Monthly_Income` (-0.35) — higher income reduces turnover risk
- `Department` (-0.32) — department membership influences retention

**Feature eliminated by L1:**
- `Annual_Bonus_Training_Hours_Interaction` (0.0) — interaction term adds no predictive value beyond individual features

### ⚠️ Suspicious Finding
`Job_Satisfaction` and `Work_Life_Balance` show large **positive** coefficients — suggesting higher values predict leaving. This is counterintuitive and may indicate reversed encoding in the pre-processed dataset. It is recommended to verify the original data encoding direction before deploying this model in production.

---

## L1 vs L2 — Key Difference

| Property | L1 (Lasso) | L2 (Ridge) |
|----------|-----------|-----------|
| Penalty term | Absolute value \|w\| | Squared value w² |
| Effect on weights | Pushes weak weights to exactly 0 | Shrinks all weights toward 0 |
| Feature selection | Yes — eliminates useless features | No — keeps all features |
| Best used when | Many useless features exist | Most features carry some signal |
| This project | Zeroed 1 feature | Zeroed 0 features |

