#!/usr/bin/env python3
"""
ETDD70 Hyperparameter Tuning
==============================
Uses GridSearchCV with stratified 5-fold CV on the final 21-feature dataset
to find optimal hyperparameters for RF and LR while guarding against overfitting.
Also tests SVM (often superior on small datasets).
"""

import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, f1_score,
                             make_scorer)
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = Path(r"c:\Users\ankit\Downloads\13332134 (1)")

# ============================================================================
# Load the final 21-feature dataset
# ============================================================================
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(BASE / "etdd70_final_21feat.csv")
feat_cols = [c for c in df.columns if c not in ("participant_id", "label")]
print(f"  Shape: {df.shape}  |  Features: {len(feat_cols)}")

X = df[feat_cols].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"  Train: {X_train_sc.shape[0]}  |  Test: {X_test_sc.shape[0]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# 1. RANDOM FOREST TUNING
# ============================================================================
print("\n" + "=" * 70)
print("1. RANDOM FOREST -- GridSearchCV")
print("=" * 70)

rf_param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [4, 6],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", 0.5],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
    refit=True
)
rf_grid.fit(X_train_sc, y_train)

print(f"  Best CV Accuracy: {rf_grid.best_score_:.4f}")
print(f"  Best params: {rf_grid.best_params_}")

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test_sc)
rf_proba = rf_best.predict_proba(X_test_sc)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"  Test Accuracy: {rf_acc:.4f}")
print(f"  Test AUC:      {rf_auc:.4f}")
print(f"  Confusion:     {confusion_matrix(y_test, rf_pred).tolist()}")
print(classification_report(y_test, rf_pred, target_names=["Non-Dys", "Dyslexic"]))

# ============================================================================
# 2. LOGISTIC REGRESSION TUNING
# ============================================================================
print("=" * 70)
print("2. LOGISTIC REGRESSION -- GridSearchCV")
print("=" * 70)

lr_param_grid = {
    "C": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "max_iter": [5000],
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
    refit=True
)
lr_grid.fit(X_train_sc, y_train)

print(f"  Best CV Accuracy: {lr_grid.best_score_:.4f}")
print(f"  Best params: {lr_grid.best_params_}")

lr_best = lr_grid.best_estimator_
lr_pred = lr_best.predict(X_test_sc)
lr_proba = lr_best.predict_proba(X_test_sc)[:, 1]
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_proba)

print(f"  Test Accuracy: {lr_acc:.4f}")
print(f"  Test AUC:      {lr_auc:.4f}")
print(f"  Confusion:     {confusion_matrix(y_test, lr_pred).tolist()}")
print(classification_report(y_test, lr_pred, target_names=["Non-Dys", "Dyslexic"]))

# Show LR active features
coefs = pd.DataFrame({
    "feature": feat_cols,
    "coef": lr_best.coef_[0]
}).sort_values("coef", key=abs, ascending=False)
active = coefs[coefs["coef"].abs() > 0.001]
print(f"  Active features ({len(active)}/{len(feat_cols)}):")
for _, r in active.iterrows():
    print(f"    {r['feature']:<30s}  {r['coef']:+.4f}")

# ============================================================================
# 3. SVM TUNING (often best for small datasets)
# ============================================================================
print("\n" + "=" * 70)
print("3. SVM (RBF Kernel) -- GridSearchCV")
print("=" * 70)

svm_param_grid = {
    "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "gamma": ["scale", "auto", 0.01, 0.05, 0.1],
    "kernel": ["rbf"],
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
    refit=True
)
svm_grid.fit(X_train_sc, y_train)

print(f"  Best CV Accuracy: {svm_grid.best_score_:.4f}")
print(f"  Best params: {svm_grid.best_params_}")

svm_best = svm_grid.best_estimator_
svm_pred = svm_best.predict(X_test_sc)
svm_proba = svm_best.predict_proba(X_test_sc)[:, 1]
svm_acc = accuracy_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_proba)

print(f"  Test Accuracy: {svm_acc:.4f}")
print(f"  Test AUC:      {svm_auc:.4f}")
print(f"  Confusion:     {confusion_matrix(y_test, svm_pred).tolist()}")
print(classification_report(y_test, svm_pred, target_names=["Non-Dys", "Dyslexic"]))

# ============================================================================
# 4. GRADIENT BOOSTING TUNING
# ============================================================================
print("=" * 70)
print("4. GRADIENT BOOSTING -- GridSearchCV")
print("=" * 70)

gb_param_grid = {
    "n_estimators": [50, 150],
    "max_depth": [2, 4],
    "learning_rate": [0.05, 0.1],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 5],
    "subsample": [0.8, 1.0],
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
    refit=True
)
gb_grid.fit(X_train_sc, y_train)

print(f"  Best CV Accuracy: {gb_grid.best_score_:.4f}")
print(f"  Best params: {gb_grid.best_params_}")

gb_best = gb_grid.best_estimator_
gb_pred = gb_best.predict(X_test_sc)
gb_proba = gb_best.predict_proba(X_test_sc)[:, 1]
gb_acc = accuracy_score(y_test, gb_pred)
gb_auc = roc_auc_score(y_test, gb_proba)

print(f"  Test Accuracy: {gb_acc:.4f}")
print(f"  Test AUC:      {gb_auc:.4f}")
print(f"  Confusion:     {confusion_matrix(y_test, gb_pred).tolist()}")
print(classification_report(y_test, gb_pred, target_names=["Non-Dys", "Dyslexic"]))

# ============================================================================
# 5. COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

results = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression", "SVM (RBF)", "Gradient Boosting"],
    "Best CV Acc": [rf_grid.best_score_, lr_grid.best_score_,
                    svm_grid.best_score_, gb_grid.best_score_],
    "Test Acc": [rf_acc, lr_acc, svm_acc, gb_acc],
    "Test AUC": [rf_auc, lr_auc, svm_auc, gb_auc],
    "Test F1": [f1_score(y_test, rf_pred), f1_score(y_test, lr_pred),
                f1_score(y_test, svm_pred), f1_score(y_test, gb_pred)],
})
results = results.sort_values("Test AUC", ascending=False).reset_index(drop=True)

print(results.to_string(index=False))

# Pick the best model (by CV accuracy to avoid test-set overfitting)
best_idx = results["Best CV Acc"].idxmax()
best_name = results.loc[best_idx, "Model"]
print(f"\n  >> Best model (by CV Accuracy): {best_name}")

# ============================================================================
# 6. SAVE THE BEST TUNED MODELS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING TUNED MODELS")
print("=" * 70)

models = {
    "Random Forest": rf_best,
    "Logistic Regression": lr_best,
    "SVM (RBF)": svm_best,
    "Gradient Boosting": gb_best,
}

for name, model in models.items():
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"tuned_{safe_name}.joblib"
    joblib.dump(model, BASE / fname)
    print(f"  Saved: {fname}")

joblib.dump(scaler, BASE / "tuned_scaler.joblib")
joblib.dump(imputer, BASE / "tuned_imputer.joblib")
print("  Saved: tuned_scaler.joblib")
print("  Saved: tuned_imputer.joblib")

# Save best params for reproducibility
best_params = {
    "random_forest": rf_grid.best_params_,
    "logistic_regression": lr_grid.best_params_,
    "svm_rbf": svm_grid.best_params_,
    "gradient_boosting": gb_grid.best_params_,
}
joblib.dump(best_params, BASE / "tuned_best_params.joblib")
print("  Saved: tuned_best_params.joblib")

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

models_list = results["Model"]
x = np.arange(len(models_list))

axes[0].barh(models_list, results["Best CV Acc"], color=["#FF6B6B","#4ECDC4","#45B7D1","#A78BFA"])
axes[0].set_xlabel("Accuracy")
axes[0].set_title("CV Accuracy (5-Fold)", fontweight="bold")
axes[0].set_xlim(0.6, 1.0)

axes[1].barh(models_list, results["Test Acc"], color=["#FF6B6B","#4ECDC4","#45B7D1","#A78BFA"])
axes[1].set_xlabel("Accuracy")
axes[1].set_title("Test Accuracy", fontweight="bold")
axes[1].set_xlim(0.6, 1.0)

axes[2].barh(models_list, results["Test AUC"], color=["#FF6B6B","#4ECDC4","#45B7D1","#A78BFA"])
axes[2].set_xlabel("AUC")
axes[2].set_title("Test ROC-AUC", fontweight="bold")
axes[2].set_xlim(0.6, 1.0)

fig.suptitle("ETDD70 -- Tuned Model Comparison (21 features)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(BASE / "tuned_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: tuned_model_comparison.png")

print("\n" + "=" * 70)
print("[DONE] Hyperparameter tuning complete.")
print("=" * 70)
