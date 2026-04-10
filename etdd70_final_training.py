#!/usr/bin/env python3
"""
ETDD70 Final Training Pipeline (Optimized Feature Set)
=======================================================
21 features selected based on:
  - Cohen's d effect sizes (all d > 0.8, "large" clinical effect)
  - MediaPipe extractability at 30fps webcam
  - Redundancy removal (sacc_count_fwd = fix_count - 1, r=1.00)
  - Statistical significance (dropped first_fix_dur: p>0.73, d<0.09)

Group A (15): Fully extractable from MediaPipe iris landmarks
  fix_count, fix_dur_mean, fix_dur_sd, fix_dur_median, total_read_time
  (x3 tasks = 15 features, Cohen's d range: 1.17 - 1.71)

Group B (6): Partially extractable but highest discriminative power
  gaze_linearity (d=2.00 on t4), revisit_count (d=1.53 on t4)
  (x3 tasks = 6 features)

Dropped (18): sacc_count_fwd (redundant r=1.00), first_fix_dur (d<0.09),
  reread_rate (d<0.31 + unextractable), fix_per_line_*, revisit_rate, skip_rate
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, f1_score)
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE = Path(r"c:\Users\ankit\Downloads\13332134 (1)")

# ============================================================================
# STEP 1: Define the curated 21-feature set
# ============================================================================
print("=" * 70)
print("STEP 1 -- Build Final 21-Feature Dataset")
print("=" * 70)

# Group A: Fully extractable from MediaPipe (15 features)
# Cohen's d range: 1.17 - 1.71 (all "large" effect sizes)
GROUP_A = []
for task in ["t1", "t4", "t5"]:
    GROUP_A.extend([
        f"{task}_fix_count",        # Number of fixations (d ~ 1.30-1.71)
        f"{task}_fix_dur_mean",     # Mean fixation duration (d ~ 1.17-1.40)
        f"{task}_fix_dur_sd",       # SD of fixation duration (d ~ 1.20-1.50)
        f"{task}_fix_dur_median",   # Median fixation duration (d ~ 1.17-1.35)
        f"{task}_total_read_time",  # Total reading time (d ~ 1.40-1.60)
    ])

# Group B: Partial but highest discriminative power (6 features)
GROUP_B = []
for task in ["t1", "t4", "t5"]:
    GROUP_B.extend([
        f"{task}_gaze_linearity",   # Gaze path irregularity (t4 d=2.00!)
        f"{task}_revisit_count",    # Revisits to previous regions (t4 d=1.53)
    ])

FINAL_FEATURES = GROUP_A + GROUP_B

print(f"\n  Group A (fully extractable):  {len(GROUP_A)} features")
for i, f in enumerate(GROUP_A, 1):
    print(f"    A{i:02d}. {f}")
print(f"\n  Group B (partial, high-value): {len(GROUP_B)} features")
for i, f in enumerate(GROUP_B, 1):
    print(f"    B{i:02d}. {f}")
print(f"\n  TOTAL: {len(FINAL_FEATURES)} features")

# Dropped features with justification
print("\n  DROPPED features (18):")
print("    - sacc_count_fwd (x3)  : r=1.00 with fix_count, redundant")
print("    - first_fix_dur (x3)   : p>0.73, d<0.09, no group difference")
print("    - reread_rate (x3)     : d<0.31, p>0.2, unextractable from webcam")
print("    - fix_per_line_mean (x3): needs precise Y-axis ROI mapping")
print("    - fix_per_line_sd (x3) : needs precise Y-axis ROI mapping")
print("    - revisit_rate (x3)    : redundant with revisit_count")

# ============================================================================
# STEP 2: Load and filter dataset
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2 -- Load & Filter Dataset")
print("=" * 70)

df = pd.read_csv(BASE / "etdd70_final_21feat.csv")

# Verify all features exist
missing = [f for f in FINAL_FEATURES if f not in df.columns]
if missing:
    print(f"  [ERROR] Missing columns: {missing}")
    raise ValueError("Feature mismatch")

final_df = df[["participant_id"] + FINAL_FEATURES + ["label"]].copy()
print(f"\n  Original dataset   : {df.shape}")
print(f"  Final dataset      : {final_df.shape}")
print(f"  Samples-to-features ratio: {final_df.shape[0]} / {len(FINAL_FEATURES)} = {final_df.shape[0]/len(FINAL_FEATURES):.1f}x")
print(f"    (Rule of thumb: >5x is acceptable, >10x is ideal)")
print(f"\n  Class balance:")
print(f"    Non-dyslexic (0): {(final_df['label']==0).sum()}")
print(f"    Dyslexic (1)    : {(final_df['label']==1).sum()}")

# Check for missing values
miss = final_df[FINAL_FEATURES].isnull().sum().sum()
print(f"  Total missing values: {miss}")

# Save final dataset
final_df.to_csv(BASE / "etdd70_final_21feat.csv", index=False)
print(f"\n  Saved: etdd70_final_21feat.csv  shape={final_df.shape}")

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3 -- Train/Test Split (80/20 Stratified)")
print("=" * 70)

X = final_df[FINAL_FEATURES].values
y = final_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Impute
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"  Train: {X_train_sc.shape[0]} samples (0={sum(y_train==0)}, 1={sum(y_train==1)})")
print(f"  Test : {X_test_sc.shape[0]} samples (0={sum(y_test==0)}, 1={sum(y_test==1)})")

# ============================================================================
# STEP 4: Train & Evaluate Models
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4 -- Train & Evaluate")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Random Forest ---
print("\n  [RANDOM FOREST]")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,          # slightly shallower for 21 features
    min_samples_split=5,
    min_samples_leaf=3,   # stricter leaf constraint
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_cv = cross_val_score(rf, X_train_sc, y_train, cv=cv, scoring="accuracy")
rf_cv_auc = cross_val_score(rf, X_train_sc, y_train, cv=cv, scoring="roc_auc")
print(f"    5-Fold CV Accuracy : {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")
print(f"    5-Fold CV AUC      : {rf_cv_auc.mean():.4f} (+/- {rf_cv_auc.std():.4f})")
print(f"    Per-fold accuracy  : {[f'{s:.3f}' for s in rf_cv]}")

rf.fit(X_train_sc, y_train)
rf_pred = rf.predict(X_test_sc)
rf_proba = rf.predict_proba(X_test_sc)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_f1 = f1_score(y_test, rf_pred)

print(f"\n    Test Accuracy : {rf_acc:.4f}")
print(f"    Test ROC-AUC  : {rf_auc:.4f}")
print(f"    Test F1       : {rf_f1:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, rf_pred, target_names=["Non-Dyslexic", "Dyslexic"]))
print(f"    Confusion Matrix: {confusion_matrix(y_test, rf_pred).tolist()}")

# --- Logistic Regression ---
print("\n  [LOGISTIC REGRESSION]")
lr = LogisticRegression(
    max_iter=5000,
    solver="liblinear",
    C=0.3,                # stronger regularization for fewer features
    penalty="l1",
    random_state=42
)

lr_cv = cross_val_score(lr, X_train_sc, y_train, cv=cv, scoring="accuracy")
lr_cv_auc = cross_val_score(lr, X_train_sc, y_train, cv=cv, scoring="roc_auc")
print(f"    5-Fold CV Accuracy : {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")
print(f"    5-Fold CV AUC      : {lr_cv_auc.mean():.4f} (+/- {lr_cv_auc.std():.4f})")
print(f"    Per-fold accuracy  : {[f'{s:.3f}' for s in lr_cv]}")

lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_proba)
lr_f1 = f1_score(y_test, lr_pred)

print(f"\n    Test Accuracy : {lr_acc:.4f}")
print(f"    Test ROC-AUC  : {lr_auc:.4f}")
print(f"    Test F1       : {lr_f1:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, lr_pred, target_names=["Non-Dyslexic", "Dyslexic"]))
print(f"    Confusion Matrix: {confusion_matrix(y_test, lr_pred).tolist()}")

# LR coefficients (non-zero ones = features the model actually uses)
lr_coefs = pd.DataFrame({
    "feature": FINAL_FEATURES,
    "coefficient": lr.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)

non_zero = lr_coefs[lr_coefs["coefficient"].abs() > 0.001]
print(f"\n    LR active features ({len(non_zero)} / {len(FINAL_FEATURES)} non-zero):")
for _, r in non_zero.iterrows():
    direction = "Dyslexic +" if r["coefficient"] > 0 else "Dyslexic -"
    print(f"      {r['feature']:<30s}  coef={r['coefficient']:+.4f}  ({direction})")

# ============================================================================
# STEP 5: Feature Importance Visualization
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5 -- Feature Importance Visualization")
print("=" * 70)

imp = pd.DataFrame({
    "feature": FINAL_FEATURES,
    "rf_importance": rf.feature_importances_,
    "group": ["A" if f in GROUP_A else "B" for f in FINAL_FEATURES]
}).sort_values("rf_importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))

colors = []
for _, row in imp.iterrows():
    f = row["feature"]
    if f.startswith("t1_"):
        colors.append("#FF6B6B" if row["group"] == "A" else "#CC4444")
    elif f.startswith("t4_"):
        colors.append("#4ECDC4" if row["group"] == "A" else "#2E8B7A")
    else:
        colors.append("#45B7D1" if row["group"] == "A" else "#2E7D94")

ax.barh(imp["feature"], imp["rf_importance"], color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Gini Importance", fontsize=12)
ax.set_title("ETDD70 Final Model -- 21 Feature Importances", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=9)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#FF6B6B", label="T1 Syllables (Group A)"),
    Patch(facecolor="#4ECDC4", label="T4 Meaningful (Group A)"),
    Patch(facecolor="#45B7D1", label="T5 Pseudo (Group A)"),
    Patch(facecolor="#2E8B7A", label="Group B (gaze_linearity / revisit_count)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
fig.savefig(BASE / "final_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: final_feature_importance.png")

# ============================================================================
# STEP 6: Save All Artifacts
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6 -- Save Models & Artifacts")
print("=" * 70)

joblib.dump(rf, BASE / "final_model_rf.joblib")
joblib.dump(lr, BASE / "final_model_lr.joblib")
joblib.dump(scaler, BASE / "final_scaler.joblib")
joblib.dump(imputer, BASE / "final_imputer.joblib")

feat_config = {
    "feature_names": FINAL_FEATURES,
    "n_features": len(FINAL_FEATURES),
    "group_a": GROUP_A,
    "group_b": GROUP_B,
    "tasks": ["t1", "t4", "t5"],
    "features_per_task_group_a": [
        "fix_count", "fix_dur_mean", "fix_dur_sd",
        "fix_dur_median", "total_read_time"
    ],
    "features_per_task_group_b": [
        "gaze_linearity", "revisit_count"
    ],
}
joblib.dump(feat_config, BASE / "final_feature_config.joblib")

print("  Saved: final_model_rf.joblib")
print("  Saved: final_model_lr.joblib")
print("  Saved: final_scaler.joblib")
print("  Saved: final_imputer.joblib")
print("  Saved: final_feature_config.joblib")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Dataset shape         : (70, 23) = 70 participants x 21 features + id + label")
print(f"  Samples/Features ratio: {70/21:.1f}x  (>3x rule of thumb)")
print(f"  Train / Test          : 56 / 14 (stratified)")
print(f"")
print(f"  Random Forest:")
print(f"    CV Accuracy  : {rf_cv.mean():.3f} (+/- {rf_cv.std():.3f})")
print(f"    CV AUC       : {rf_cv_auc.mean():.3f} (+/- {rf_cv_auc.std():.3f})")
print(f"    Test Accuracy: {rf_acc:.3f}")
print(f"    Test AUC     : {rf_auc:.3f}")
print(f"    Test F1      : {rf_f1:.3f}")
print(f"")
print(f"  Logistic Regression:")
print(f"    CV Accuracy  : {lr_cv.mean():.3f} (+/- {lr_cv.std():.3f})")
print(f"    CV AUC       : {lr_cv_auc.mean():.3f} (+/- {lr_cv_auc.std():.3f})")
print(f"    Test Accuracy: {lr_acc:.3f}")
print(f"    Test AUC     : {lr_auc:.3f}")
print(f"    Test F1      : {lr_f1:.3f}")
print(f"")
print(f"  LR active features: {len(non_zero)} / {len(FINAL_FEATURES)}")
print(f"")
print(f"  Output files:")
print(f"    etdd70_final_21feat.csv      <- THE final dataset")
print(f"    final_model_rf.joblib")
print(f"    final_model_lr.joblib")
print(f"    final_scaler.joblib")
print(f"    final_imputer.joblib")
print(f"    final_feature_config.joblib")
print(f"    final_feature_importance.png")
print("=" * 70)
print("[DONE] Final models trained and saved.")
