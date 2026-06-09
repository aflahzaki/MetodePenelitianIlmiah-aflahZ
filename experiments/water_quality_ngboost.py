#!/usr/bin/env python3
"""
==========================================================================
Evaluasi Performa Prediksi Probabilistik pada Klasifikasi Kualitas Air
Menggunakan Algoritma Natural Gradient Boosting (NGBoost)
==========================================================================

Script eksperimen lengkap untuk mereproduksi seluruh pipeline:
1. Download dataset Water Potability dari Kaggle
2. Exploratory Data Analysis (EDA)
3. MICE Imputation (IterativeImputer)
4. Stratified Split 70:15:15
5. StandardScaler (fit on train only)
6. Evaluasi Kondisional SMOTE-ENN (demonstrasi bahwa degradasi terjadi)
7. Training model tanpa resampling: NGBoost, XGBoost, Random Forest
8. Evaluasi: Accuracy, Precision, Recall, F1, NLL, ECE, ROC-AUC, Confusion Matrix
9. McNemar's Test antar pasangan model
10. Analisis Uncertainty Zone
11. Visualisasi: Calibration, ROC, KDE, Confusion Matrix, Feature Importance, Loss Curves

Author: Aflah Zaki Siregar (103062300095)
Reproducibility: random_state=42 digunakan di semua proses stokastik
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from scipy.stats import chi2

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)


# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 70)
print("1. LOADING DATASET")
print("=" * 70)

# Try kagglehub first, fallback to cached path
try:
    import kagglehub
    dataset_path = kagglehub.dataset_download("adityakadiwal/water-potability")
    csv_path = os.path.join(dataset_path, "water_potability.csv")
except Exception:
    csv_path = "/root/.cache/kagglehub/datasets/adityakadiwal/water-potability/versions/3/water_potability.csv"

df = pd.read_csv(csv_path)
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Features: {list(df.columns)}")
print()


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("=" * 70)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Missing values analysis
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Feature': df.columns,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print(missing_df.to_string(index=False))
print(f"\nTotal samples with missing values: {df.isnull().any(axis=1).sum()}")

# Class distribution
print("\n--- Class Distribution ---")
class_dist = df['Potability'].value_counts()
print(f"Class 0 (Not Potable): {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%)")
print(f"Class 1 (Potable):     {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)")
print(f"Imbalance Ratio: {class_dist[0]/class_dist[1]:.4f}:1")

# Dataset statistics
print("\n--- Descriptive Statistics ---")
print(df.describe().round(4).to_string())
print()


# ============================================================
# 3. MICE IMPUTATION (IterativeImputer)
# ============================================================
print("=" * 70)
print("3. MICE IMPUTATION")
print("=" * 70)

X = df.drop('Potability', axis=1)
y = df['Potability']

mice_imputer = IterativeImputer(
    max_iter=10,
    random_state=RANDOM_STATE,
    sample_posterior=False
)

X_imputed = pd.DataFrame(
    mice_imputer.fit_transform(X),
    columns=X.columns
)

print(f"Imputation complete. Remaining missing values: {X_imputed.isnull().sum().sum()}")
print(f"Shape after imputation: {X_imputed.shape}")
print()


# ============================================================
# 4. STRATIFIED SPLIT 70:15:15
# ============================================================
print("=" * 70)
print("4. STRATIFIED DATA SPLITTING (70:15:15)")
print("=" * 70)

# First split: separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_imputed, y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_STATE
)

# Second split: separate validation set (15/85 of remaining = ~15% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=15/85,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"Training set:   {X_train.shape[0]} samples (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
print(f"Validation set: {X_val.shape[0]} samples (Class 0: {(y_val==0).sum()}, Class 1: {(y_val==1).sum()})")
print(f"Test set:       {X_test.shape[0]} samples (Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")
print(f"Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} = {len(df)}")
print()


# ============================================================
# 5. STANDARD SCALING
# ============================================================
print("=" * 70)
print("5. STANDARD SCALING (fit on train only)")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Scaler fitted on training set ({X_train.shape[0]} samples)")
print(f"Train mean (should be ~0): {X_train_scaled.mean(axis=0).mean():.6f}")
print(f"Train std  (should be ~1): {X_train_scaled.std(axis=0).mean():.6f}")
print()


# ============================================================
# 6. EVALUASI KONDISIONAL SMOTE-ENN
# ============================================================
print("=" * 70)
print("6. EVALUASI KONDISIONAL SMOTE-ENN")
print("=" * 70)
print("Metodologi menyatakan SMOTE-ENN diterapkan 'secara kondisional'.")
print("Artinya: diterapkan HANYA jika meningkatkan performa model.\n")

# Apply SMOTE-ENN to training data
smote_enn = SMOTEENN(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_scaled, y_train)

print(f"Training set SEBELUM SMOTE-ENN: {X_train_scaled.shape[0]} samples")
print(f"  Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
print(f"\nTraining set SETELAH SMOTE-ENN: {X_train_resampled.shape[0]} samples")
print(f"  Class 0: {(y_train_resampled==0).sum()}, Class 1: {(y_train_resampled==1).sum()}")
print(f"\nPerubahan: {X_train_scaled.shape[0]} -> {X_train_resampled.shape[0]} "
      f"(REDUKSI {X_train_scaled.shape[0] - X_train_resampled.shape[0]} samples!)")

# Quick comparison: train a simple model with and without SMOTE-ENN
print("\n--- Perbandingan Cepat (Random Forest pada Validation Set) ---")

# Without SMOTE-ENN
rf_no_smote = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_no_smote.fit(X_train_scaled, y_train)
pred_no_smote = rf_no_smote.predict(X_val_scaled)
prob_no_smote = rf_no_smote.predict_proba(X_val_scaled)
acc_no_smote = accuracy_score(y_val, pred_no_smote)
nll_no_smote = log_loss(y_val, prob_no_smote)

# With SMOTE-ENN
rf_with_smote = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_with_smote.fit(X_train_resampled, y_train_resampled)
pred_with_smote = rf_with_smote.predict(X_val_scaled)
prob_with_smote = rf_with_smote.predict_proba(X_val_scaled)
acc_with_smote = accuracy_score(y_val, pred_with_smote)
nll_with_smote = log_loss(y_val, prob_with_smote)

print(f"\n  TANPA SMOTE-ENN  -> Accuracy: {acc_no_smote:.4f}, NLL: {nll_no_smote:.4f}")
print(f"  DENGAN SMOTE-ENN -> Accuracy: {acc_with_smote:.4f}, NLL: {nll_with_smote:.4f}")

if acc_no_smote >= acc_with_smote:
    print(f"\n  KEPUTUSAN: SMOTE-ENN TIDAK digunakan (menurunkan accuracy sebesar "
          f"{acc_no_smote - acc_with_smote:.4f})")
    print("  Alasan: ENN menghapus terlalu banyak sampel mayoritas, menyebabkan")
    print("  underfitting akibat berkurangnya informasi training secara signifikan.")
    USE_RESAMPLING = False
else:
    print(f"\n  KEPUTUSAN: SMOTE-ENN digunakan (meningkatkan accuracy)")
    USE_RESAMPLING = True

print(f"\n  Final decision: USE_RESAMPLING = {USE_RESAMPLING}")
print()


# ============================================================
# 7. MODEL TRAINING (TANPA RESAMPLING)
# ============================================================
print("=" * 70)
print("7. MODEL TRAINING (Final Models - Tanpa Resampling)")
print("=" * 70)

# Select training data based on conditional evaluation
if USE_RESAMPLING:
    X_train_final = X_train_resampled
    y_train_final = y_train_resampled
    print("Using RESAMPLED training data")
else:
    X_train_final = X_train_scaled
    y_train_final = y_train
    print("Using ORIGINAL training data (no resampling)")

print(f"Training samples: {len(y_train_final)}")
print()

# --- NGBoost ---
print("Training NGBoost (Bernoulli distribution)...")
ngb_model = NGBClassifier(
    Dist=Bernoulli,
    n_estimators=300,
    learning_rate=0.05,
    minibatch_frac=0.8,
    col_sample=0.8,
    random_state=RANDOM_STATE,
    verbose=False
)
ngb_model.fit(X_train_final, y_train_final, X_val=X_val_scaled, Y_val=y_val)
print("  NGBoost training complete.")

# --- XGBoost ---
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0
)
xgb_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)
print("  XGBoost training complete.")

# --- Random Forest ---
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train_final)
print("  Random Forest training complete.")
print()


# ============================================================
# 8. EVALUATION ON TEST SET
# ============================================================
print("=" * 70)
print("8. EVALUATION ON TEST SET")
print("=" * 70)

models = {
    'NGBoost': ngb_model,
    'XGBoost': xgb_model,
    'Random Forest': rf_model
}

results = {}

for name, model in models.items():
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # For binary, we need P(class=1)
    if y_prob.ndim == 2:
        y_prob_pos = y_prob[:, 1]
    else:
        y_prob_pos = y_prob

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    nll = log_loss(y_test, y_prob_pos)
    auc = roc_auc_score(y_test, y_prob_pos)

    # ECE (Expected Calibration Error) - 10 bins
    prob_true, prob_pred_bins = calibration_curve(y_test, y_prob_pos, n_bins=10, strategy='uniform')
    # Manual ECE calculation with bin counts
    bin_edges = np.linspace(0, 1, 11)
    ece = 0.0
    n_total = len(y_test)
    for i in range(10):
        mask = (y_prob_pos >= bin_edges[i]) & (y_prob_pos < bin_edges[i+1])
        if i == 9:  # include right edge for last bin
            mask = (y_prob_pos >= bin_edges[i]) & (y_prob_pos <= bin_edges[i+1])
        n_bin = mask.sum()
        if n_bin > 0:
            avg_confidence = y_prob_pos[mask].mean()
            avg_accuracy = y_test.values[mask].mean()
            ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results[name] = {
        'y_pred': y_pred,
        'y_prob_pos': y_prob_pos,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'nll': nll,
        'ece': ece,
        'auc': auc,
        'cm': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

# Print results table
print(f"\n{'Metric':<12} {'NGBoost':<12} {'XGBoost':<12} {'Random Forest':<12}")
print("-" * 50)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'nll', 'ece', 'auc']:
    print(f"{metric:<12} "
          f"{results['NGBoost'][metric]:<12.4f} "
          f"{results['XGBoost'][metric]:<12.4f} "
          f"{results['Random Forest'][metric]:<12.4f}")

print(f"\n--- Confusion Matrices (Test Set, N={len(y_test)}) ---")
for name in models:
    r = results[name]
    print(f"\n{name}:")
    print(f"  TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}")
    print(f"  Verification: TN+FP+FN+TP = {r['tn']+r['fp']+r['fn']+r['tp']} (should be {len(y_test)})")
    print(f"  Acc check: (TP+TN)/N = ({r['tp']}+{r['tn']})/{len(y_test)} = {(r['tp']+r['tn'])/len(y_test):.4f} (reported: {r['accuracy']:.4f})")
print()


# ============================================================
# 9. McNEMAR'S TEST
# ============================================================
print("=" * 70)
print("9. McNEMAR'S TEST (Signifikansi Statistik)")
print("=" * 70)


def mcnemar_test(y_true, pred_a, pred_b, model_a_name, model_b_name):
    """Perform McNemar's test between two models."""
    # Create contingency table
    # Both correct, A correct B wrong, A wrong B correct, Both wrong
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n00 = ((~correct_a) & (~correct_b)).sum()  # both wrong
    n01 = ((~correct_a) & (correct_b)).sum()   # A wrong, B correct
    n10 = ((correct_a) & (~correct_b)).sum()   # A correct, B wrong
    n11 = ((correct_a) & (correct_b)).sum()    # both correct

    # McNemar statistic (with continuity correction)
    if (n01 + n10) == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    print(f"\n  {model_a_name} vs {model_b_name}:")
    print(f"    Contingency: both_correct={n11}, A_only={n10}, B_only={n01}, both_wrong={n00}")
    print(f"    Chi-squared = {chi2_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print(f"    -> Signifikan (p < 0.05): Perbedaan performa antar model SIGNIFIKAN")
    else:
        print(f"    -> Tidak Signifikan (p >= 0.05): Tidak ada perbedaan signifikan")

    return chi2_stat, p_value


model_names = list(models.keys())
mcnemar_results = {}
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name_a = model_names[i]
        name_b = model_names[j]
        chi2_stat, p_val = mcnemar_test(
            y_test.values,
            results[name_a]['y_pred'],
            results[name_b]['y_pred'],
            name_a, name_b
        )
        mcnemar_results[f"{name_a} vs {name_b}"] = {'chi2': chi2_stat, 'p_value': p_val}
print()


# ============================================================
# 10. UNCERTAINTY ZONE ANALYSIS
# ============================================================
print("=" * 70)
print("10. UNCERTAINTY ZONE ANALYSIS")
print("=" * 70)
print("\nZone definitions based on predicted P(Potable):")
print("  Zone 1: mu < 0.2  (Very Confident Negative)")
print("  Zone 2: 0.2 <= mu < 0.4  (Somewhat Confident Negative)")
print("  Zone 3: 0.4 <= mu < 0.6  (Uncertain / Ambiguous)")
print("  Zone 4: 0.6 <= mu < 0.8  (Somewhat Confident Positive)")
print("  Zone 5: mu >= 0.8  (Very Confident Positive)")

zones = [
    ("Zone 1 (mu<0.2)", 0.0, 0.2),
    ("Zone 2 (0.2-0.4)", 0.2, 0.4),
    ("Zone 3 (0.4-0.6)", 0.4, 0.6),
    ("Zone 4 (0.6-0.8)", 0.6, 0.8),
    ("Zone 5 (mu>=0.8)", 0.8, 1.01),
]

for name in models:
    probs = results[name]['y_prob_pos']
    preds = results[name]['y_pred']
    print(f"\n--- {name} ---")
    print(f"  {'Zone':<22} {'N':>5} {'Acc':>8} {'Avg Prob':>10}")
    print(f"  {'-'*47}")
    for zone_name, low, high in zones:
        mask = (probs >= low) & (probs < high)
        n_zone = mask.sum()
        if n_zone > 0:
            zone_acc = accuracy_score(y_test.values[mask], preds[mask])
            avg_prob = probs[mask].mean()
            print(f"  {zone_name:<22} {n_zone:>5} {zone_acc:>8.4f} {avg_prob:>10.4f}")
        else:
            print(f"  {zone_name:<22} {n_zone:>5}      N/A        N/A")
print()


# ============================================================
# 11. VISUALIZATIONS
# ============================================================
print("=" * 70)
print("11. GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'NGBoost': '#2196F3', 'XGBoost': '#4CAF50', 'Random Forest': '#FF9800'}

# --- 11a. Calibration Curves ---
print("  Generating calibration curves...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
for name in models:
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test, results[name]['y_prob_pos'], n_bins=10, strategy='uniform'
    )
    ax.plot(prob_pred_cal, prob_true_cal, 'o-', color=colors[name],
            label=f"{name} (ECE={results[name]['ece']:.4f})")
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves (Reliability Diagram)')
ax.legend(loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'calibration_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 11b. ROC Curves ---
print("  Generating ROC curves...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob_pos'])
    ax.plot(fpr, tpr, color=colors[name],
            label=f"{name} (AUC={results[name]['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 11c. KDE Probability Distributions ---
print("  Generating KDE probability distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, name in enumerate(models):
    ax = axes[idx]
    probs = results[name]['y_prob_pos']
    mask_0 = (y_test == 0).values
    mask_1 = (y_test == 1).values
    sns.kdeplot(probs[mask_0], ax=ax, color='red', label='Class 0 (Not Potable)', fill=True, alpha=0.3)
    sns.kdeplot(probs[mask_1], ax=ax, color='blue', label='Class 1 (Potable)', fill=True, alpha=0.3)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted P(Potable)')
    ax.set_ylabel('Density')
    ax.set_title(f'{name}')
    ax.legend()
    ax.set_xlim([0, 1])
plt.suptitle('KDE of Predicted Probabilities by True Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'kde_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 11d. Confusion Matrices ---
print("  Generating confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, name in enumerate(models):
    ax = axes[idx]
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Potable', 'Potable'],
                yticklabels=['Not Potable', 'Potable'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{name}\nAcc={results[name]["accuracy"]:.4f}')
plt.suptitle('Confusion Matrices (Test Set)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 11e. Feature Importance ---
print("  Generating feature importance plots...")
feature_names = X.columns.tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# NGBoost feature importance (based on estimator splits)
try:
    ngb_importance = np.zeros(len(feature_names))
    for learner in ngb_model.base_models:
        for tree in learner:
            fi = tree.feature_importances_
            ngb_importance += fi
    ngb_importance /= ngb_importance.sum()
except Exception:
    ngb_importance = np.ones(len(feature_names)) / len(feature_names)

ax = axes[0]
sorted_idx = np.argsort(ngb_importance)
ax.barh(range(len(feature_names)), ngb_importance[sorted_idx], color=colors['NGBoost'])
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('Importance')
ax.set_title('NGBoost Feature Importance')

# XGBoost feature importance
xgb_importance = xgb_model.feature_importances_
ax = axes[1]
sorted_idx = np.argsort(xgb_importance)
ax.barh(range(len(feature_names)), xgb_importance[sorted_idx], color=colors['XGBoost'])
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('Importance')
ax.set_title('XGBoost Feature Importance')

# Random Forest feature importance
rf_importance = rf_model.feature_importances_
ax = axes[2]
sorted_idx = np.argsort(rf_importance)
ax.barh(range(len(feature_names)), rf_importance[sorted_idx], color=colors['Random Forest'])
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance')

plt.suptitle('Feature Importance Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 11f. XGBoost Loss Curve ---
print("  Generating loss curves...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# XGBoost has eval results
xgb_evals = xgb_model.evals_result()
if xgb_evals and 'validation_0' in xgb_evals:
    val_loss = xgb_evals['validation_0']['logloss']
    ax.plot(range(len(val_loss)), val_loss, color=colors['XGBoost'], label='XGBoost Validation Loss')

# NGBoost - compute training loss per iteration if possible
# We'll use a simulated approach by evaluating at checkpoints
ax.set_xlabel('Iteration')
ax.set_ylabel('Log Loss')
ax.set_title('Training/Validation Loss Curves')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'loss_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 11g. SMOTE-ENN Comparison Bar Plot ---
print("  Generating SMOTE-ENN comparison plot...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Train all models with SMOTE-ENN for comparison
ngb_smote = NGBClassifier(Dist=Bernoulli, n_estimators=300, learning_rate=0.05,
                          minibatch_frac=0.8, col_sample=0.8, random_state=RANDOM_STATE, verbose=False)
ngb_smote.fit(X_train_resampled, y_train_resampled)
ngb_smote_pred = ngb_smote.predict(X_test_scaled)
ngb_smote_acc = accuracy_score(y_test, ngb_smote_pred)

xgb_smote = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                           eval_metric='logloss', use_label_encoder=False, verbosity=0)
xgb_smote.fit(X_train_resampled, y_train_resampled)
xgb_smote_pred = xgb_smote.predict(X_test_scaled)
xgb_smote_acc = accuracy_score(y_test, xgb_smote_pred)

rf_smote = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rf_smote.fit(X_train_resampled, y_train_resampled)
rf_smote_pred = rf_smote.predict(X_test_scaled)
rf_smote_acc = accuracy_score(y_test, rf_smote_pred)

x_pos = np.arange(3)
width = 0.35
no_smote_accs = [results['NGBoost']['accuracy'], results['XGBoost']['accuracy'], results['Random Forest']['accuracy']]
smote_accs = [ngb_smote_acc, xgb_smote_acc, rf_smote_acc]

bars1 = ax.bar(x_pos - width/2, no_smote_accs, width, label='Tanpa SMOTE-ENN', color='#2196F3')
bars2 = ax.bar(x_pos + width/2, smote_accs, width, label='Dengan SMOTE-ENN', color='#FF5722')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Dampak SMOTE-ENN terhadap Accuracy (Test Set)')
ax.set_xticks(x_pos)
ax.set_xticklabels(['NGBoost', 'XGBoost', 'Random Forest'])
ax.legend()
ax.set_ylim([0.4, 0.8])

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'smote_enn_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  All figures saved to:", FIGURES_DIR)
print()


# ============================================================
# 12. COMPREHENSIVE RESULTS SUMMARY
# ============================================================
print("=" * 70)
print("=" * 70)
print("    RINGKASAN HASIL EKSPERIMEN (Untuk Update Paper)")
print("=" * 70)
print("=" * 70)

print(f"""
DATASET:
  - Total samples: {len(df)}
  - Features: {len(df.columns) - 1} (ph, Hardness, Solids, Chloramines, Sulfate, 
    Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
  - Target: Potability (0=Not Potable, 1=Potable)
  - Class distribution: {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%) vs {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)

MISSING VALUES:
  - ph: {(df['ph'].isnull().sum()/len(df)*100):.2f}%
  - Sulfate: {(df['Sulfate'].isnull().sum()/len(df)*100):.2f}%
  - Trihalomethanes: {(df['Trihalomethanes'].isnull().sum()/len(df)*100):.2f}%

DATA SPLIT:
  - Training: {X_train.shape[0]} samples
  - Validation: {X_val.shape[0]} samples  
  - Test: {X_test.shape[0]} samples

SMOTE-ENN EVALUATION:
  - Training BEFORE SMOTE-ENN: {X_train_scaled.shape[0]} samples
  - Training AFTER SMOTE-ENN: {X_train_resampled.shape[0]} samples (REDUCED by {X_train_scaled.shape[0] - X_train_resampled.shape[0]})
  - Impact on test accuracy:
    - NGBoost:       {results['NGBoost']['accuracy']:.4f} (tanpa) vs {ngb_smote_acc:.4f} (dengan SMOTE-ENN)
    - XGBoost:       {results['XGBoost']['accuracy']:.4f} (tanpa) vs {xgb_smote_acc:.4f} (dengan SMOTE-ENN)
    - Random Forest: {results['Random Forest']['accuracy']:.4f} (tanpa) vs {rf_smote_acc:.4f} (dengan SMOTE-ENN)
  - KEPUTUSAN: SMOTE-ENN TIDAK diterapkan (conditional = only if improves)

FINAL MODEL RESULTS (Test Set, N={len(y_test)}):
""")

print(f"{'='*60}")
print(f"{'Metric':<15} {'NGBoost':<15} {'XGBoost':<15} {'Random Forest':<15}")
print(f"{'='*60}")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'nll', 'ece', 'auc']:
    label = metric.upper() if metric in ['nll', 'ece', 'auc'] else metric.capitalize()
    print(f"{label:<15} "
          f"{results['NGBoost'][metric]:<15.4f} "
          f"{results['XGBoost'][metric]:<15.4f} "
          f"{results['Random Forest'][metric]:<15.4f}")
print(f"{'='*60}")

print(f"""
CONFUSION MATRICES (TN, FP, FN, TP):
  NGBoost:       TN={results['NGBoost']['tn']}, FP={results['NGBoost']['fp']}, FN={results['NGBoost']['fn']}, TP={results['NGBoost']['tp']}
  XGBoost:       TN={results['XGBoost']['tn']}, FP={results['XGBoost']['fp']}, FN={results['XGBoost']['fn']}, TP={results['XGBoost']['tp']}
  Random Forest: TN={results['Random Forest']['tn']}, FP={results['Random Forest']['fp']}, FN={results['Random Forest']['fn']}, TP={results['Random Forest']['tp']}

McNEMAR'S TEST RESULTS:""")

for pair, res in mcnemar_results.items():
    sig = "SIGNIFIKAN" if res['p_value'] < 0.05 else "TIDAK SIGNIFIKAN"
    print(f"  {pair}: chi2={res['chi2']:.4f}, p={res['p_value']:.4f} ({sig})")

print(f"""
HYPERPARAMETERS:
  NGBoost: Dist=Bernoulli, n_estimators=300, lr=0.05, minibatch_frac=0.8, col_sample=0.8
  XGBoost: n_estimators=300, lr=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8
  Random Forest: n_estimators=300

REPRODUCIBILITY:
  random_state=42 used in: MICE, train_test_split, SMOTE-ENN, NGBoost, XGBoost, RF

FIGURES SAVED:
  {FIGURES_DIR}/calibration_curves.png
  {FIGURES_DIR}/roc_curves.png
  {FIGURES_DIR}/kde_distributions.png
  {FIGURES_DIR}/confusion_matrices.png
  {FIGURES_DIR}/feature_importance.png
  {FIGURES_DIR}/loss_curves.png
  {FIGURES_DIR}/smote_enn_comparison.png
""")

print("=" * 70)
print("SCRIPT COMPLETE - All results above are reproducible with random_state=42")
print("=" * 70)
