


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve
)

print("=" * 60)
print("  PsyScreening — Mental Health Screening ML Pipeline")
print("=" * 60)


#  LOAD DATA

DATA_PATH = "survey.csv"   # ganti path jika perlu

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Data Loaded : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Target       : treatment — {df['treatment'].value_counts().to_dict()}")



# EXPLORATORY DATA ANALYSIS (EDA)

print("\n[2] EDA Summary")
print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")
print(f"\n    Age stats: min={df['Age'].min()}, max={df['Age'].max()}, "
      f"mean={df['Age'].mean():.1f}")


# DATA CLEANING

print("\n[3] Cleaning Data...")

df.drop(columns=['Timestamp', 'state', 'comments'], inplace=True)

before = len(df)
df = df[(df['Age'] >= 15) & (df['Age'] <= 75)].copy()
print(f"    Age filter: {before} -> {len(df)} rows (removed {before-len(df)} outliers)")

def normalize_gender(g):
    g = str(g).lower().strip()
    if any(k in g for k in ['male', 'man', 'maile', 'cis male']): return 'Male'
    if any(k in g for k in ['female', 'woman', 'femail', 'femake']): return 'Female'
    return 'Other'

df['Gender'] = df['Gender'].apply(normalize_gender)
print(f"    Gender normalized: {df['Gender'].value_counts().to_dict()}")

df['self_employed'].fillna('No', inplace=True)
df['work_interfere'].fillna('Unknown', inplace=True)
print("    Missing values handled.")


# FEATURE ENGINEERING

print("\n[4] Feature Engineering...")

df['treatment'] = (df['treatment'] == 'Yes').astype(int)

ordinal_maps = {
    'work_interfere': {
        'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Unknown': 1
    },
    'leave': {
        'Very easy': 0, 'Somewhat easy': 1, 'Unknown': 2,
        'Somewhat difficult': 3, 'Very difficult': 4
    },
    'no_employees': {
        '1-5': 0, '6-25': 1, '26-100': 2,
        '100-500': 3, '500-1000': 4, 'More than 1000': 5
    },
}

X = df.drop(columns=['treatment', 'Country'])
y = df['treatment']

for col, mapping in ordinal_maps.items():
    X[col] = X[col].map(mapping).fillna(1)

le = LabelEncoder()
for c in list(X.select_dtypes(include='object').columns):
    X[c] = le.fit_transform(X[c].astype(str))

print(f"    Total features : {X.shape[1]}")
print(f"    Features       : {list(X.columns)}")


# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[5] Split: Train={len(X_train)}, Test={len(X_test)}")

scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_train)
Xte_sc = scaler.transform(X_test)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# 6. BASELINE TRAINING

print("\n[6] Baseline Training (3-Fold Stratified CV)")

baseline_configs = [
    ('Logistic Regression', LogisticRegression(max_iter=300, C=1, random_state=42), Xtr_sc),
    ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    ('Gradient Boosting',   GradientBoostingClassifier(n_estimators=100, random_state=42), X_train),
]

for name, model, Xu in baseline_configs:
    scores = cross_val_score(model, Xu, y_train, cv=cv, scoring='f1')
    print(f"    {name:<25} CV F1 = {scores.mean():.4f} +/- {scores.std():.4f}")



# 7. HYPERPARAMETER TUNING (Manual Grid Search)

print("\n[7] Hyperparameter Tuning...")

print("    -> Random Forest")
best_rf_score, best_rf_params = 0, {}
for n in [100, 200, 300]:
    for md in [None, 10, 15]:
        for mf in ['sqrt', 'log2']:
            mdl = RandomForestClassifier(
                n_estimators=n, max_depth=md, max_features=mf, random_state=42
            )
            s = cross_val_score(mdl, X_train, y_train, cv=cv, scoring='f1').mean()
            if s > best_rf_score:
                best_rf_score = s
                best_rf_params = {'n_estimators': n, 'max_depth': md, 'max_features': mf}

best_rf = RandomForestClassifier(**best_rf_params, random_state=42)
best_rf.fit(X_train, y_train)
rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]
rf_auc  = roc_auc_score(y_test, rf_prob)
rf_f1   = f1_score(y_test, rf_pred)
print(f"       Best CV F1={best_rf_score:.4f} | Test F1={rf_f1:.4f} | AUC={rf_auc:.4f}")
print(f"       Params: {best_rf_params}")

# ── Gradient Boosting ────────────────────────
print("    -> Gradient Boosting")
best_gb_score, best_gb_params = 0, {}
for n in [100, 200]:
    for lr2 in [0.05, 0.1, 0.2]:
        for md in [3, 4, 5]:
            mdl = GradientBoostingClassifier(
                n_estimators=n, learning_rate=lr2, max_depth=md, random_state=42
            )
            s = cross_val_score(mdl, X_train, y_train, cv=cv, scoring='f1').mean()
            if s > best_gb_score:
                best_gb_score = s
                best_gb_params = {'n_estimators': n, 'learning_rate': lr2, 'max_depth': md}

best_gb = GradientBoostingClassifier(**best_gb_params, random_state=42)
best_gb.fit(X_train, y_train)
gb_pred = best_gb.predict(X_test)
gb_prob = best_gb.predict_proba(X_test)[:, 1]
gb_auc  = roc_auc_score(y_test, gb_prob)
gb_f1   = f1_score(y_test, gb_pred)
print(f"       Best CV F1={best_gb_score:.4f} | Test F1={gb_f1:.4f} | AUC={gb_auc:.4f}")
print(f"       Params: {best_gb_params}")

# ── Logistic Regression ──────────────────────
print("    -> Logistic Regression")
best_lr_score, best_lr_params = 0, {}
for c in [0.01, 0.1, 1, 10, 100]:
    for pen in ['l1', 'l2']:
        mdl = LogisticRegression(
            C=c, penalty=pen, solver='liblinear', max_iter=500, random_state=42
        )
        s = cross_val_score(mdl, Xtr_sc, y_train, cv=cv, scoring='f1').mean()
        if s > best_lr_score:
            best_lr_score = s
            best_lr_params = {'C': c, 'penalty': pen}

best_lr = LogisticRegression(
    **best_lr_params, solver='liblinear', max_iter=500, random_state=42
)
best_lr.fit(Xtr_sc, y_train)
lr_pred = best_lr.predict(Xte_sc)
lr_prob = best_lr.predict_proba(Xte_sc)[:, 1]
lr_auc  = roc_auc_score(y_test, lr_prob)
lr_f1   = f1_score(y_test, lr_pred)
print(f"       Best CV F1={best_lr_score:.4f} | Test F1={lr_f1:.4f} | AUC={lr_auc:.4f}")
print(f"       Params: {best_lr_params}")


# ──────────────────────────────────────────────
# 8. SELECT CHAMPION MODEL

print("\n[8] Selecting Champion Model...")

aucs = {
    'Random Forest':       rf_auc,
    'Gradient Boosting':   gb_auc,
    'Logistic Regression': lr_auc,
}
champ_name = max(aucs, key=aucs.get)

champ_map = {
    'Random Forest':       (best_rf, X_test, False, rf_pred, rf_prob, best_rf_params),
    'Gradient Boosting':   (best_gb, X_test, False, gb_pred, gb_prob, best_gb_params),
    'Logistic Regression': (best_lr, Xte_sc, True,  lr_pred, lr_prob, best_lr_params),
}
champ_mdl, X_te, use_scaled, y_pred_c, y_prob_c, champ_params = champ_map[champ_name]

print(f"\n  Champion  : {champ_name}")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_c):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_pred_c):.4f}")
print(f"  ROC-AUC   : {aucs[champ_name]:.4f}")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, y_pred_c,
      target_names=['No Treatment', 'Needs Treatment']))



# 9. VISUALISASI HASIL

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('PsyScreening — Model Evaluation', fontsize=15, fontweight='bold')

model_names = list(aucs.keys())
auc_vals    = [aucs[m] for m in model_names]
colors      = ['#4e79a7', '#f28e2b', '#e15759']
bars = axes[0].barh(model_names, auc_vals, color=colors, edgecolor='white')
axes[0].set_xlim(0.80, 0.90)
axes[0].set_xlabel('ROC-AUC')
axes[0].set_title('Model Comparison (AUC)')
for bar, val in zip(bars, auc_vals):
    axes[0].text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=10)

cm = confusion_matrix(y_test, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['No Treatment', 'Needs Treatment'],
            yticklabels=['No Treatment', 'Needs Treatment'])
axes[1].set_title(f'Confusion Matrix\n({champ_name})')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

for nm, prob, col in [
    ('Random Forest',       rf_prob, '#4e79a7'),
    ('Gradient Boosting',   gb_prob, '#f28e2b'),
    ('Logistic Regression', lr_prob, '#e15759'),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[2].plot(fpr, tpr, label=f'{nm} (AUC={aucs[nm]:.4f})', color=col, lw=2)
axes[2].plot([0,1],[0,1],'--', color='gray', lw=1)
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curve')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('psyscreening_evaluation.png', dpi=150, bbox_inches='tight')
print("\n  Plot saved: psyscreening_evaluation.png")


# ──────────────────────────────────────────────
# 10. SAVE MODEL & METADATA

print("\n[10] Saving artifacts...")

os.makedirs('outputs', exist_ok=True)
joblib.dump(champ_mdl, 'outputs/psyscreening_model.pkl')
joblib.dump(scaler,    'outputs/psyscreening_scaler.pkl')

metadata = {
    'champion_model'    : champ_name,
    'feature_columns'   : list(X.columns),
    'target'            : 'treatment',
    'test_accuracy'     : round(accuracy_score(y_test, y_pred_c), 4),
    'test_f1'           : round(f1_score(y_test, y_pred_c), 4),
    'test_auc'          : round(aucs[champ_name], 4),
    'best_params'       : champ_params,
    'use_scaled_input'  : use_scaled,
    'all_model_aucs'    : {k: round(v, 4) for k, v in aucs.items()},
    'tuned_cv_f1'       : {
        'RF': round(best_rf_score, 4),
        'GB': round(best_gb_score, 4),
        'LR': round(best_lr_score, 4),
    },
    'train_size'        : int(len(X_train)),
    'test_size'         : int(len(X_test)),
    'feature_count'     : int(X.shape[1]),
    'notes'             : 'Age 15-75, Gender Male/Female/Other, Country dropped',
}
with open('outputs/psyscreening_meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  Saved: outputs/psyscreening_model.pkl")
print("  Saved: outputs/psyscreening_scaler.pkl")
print("  Saved: outputs/psyscreening_meta.json")



# 11. INFERENCE HELPER

def predict_single(input_dict: dict) -> dict:
    """
    Prediksi untuk 1 responden baru.

    Parameters
    ----------
    input_dict : dict — field sesuai feature_columns di metadata

    Returns
    -------
    dict : {'label': str, 'probability': float, 'needs_treatment': bool}
    """
    row = pd.DataFrame([input_dict])

    for col, mapping in ordinal_maps.items():
        if col in row.columns:
            row[col] = row[col].map(mapping).fillna(1)

    _le = LabelEncoder()
    for c in list(row.select_dtypes(include='object').columns):
        _le.fit(list(X[c].unique()))
        try:
            row[c] = _le.transform(row[c].astype(str))
        except ValueError:
            row[c] = 0

    row = row.reindex(columns=X.columns, fill_value=0)
    row_input = scaler.transform(row) if use_scaled else row.values

    pred = champ_mdl.predict(row_input)[0]
    prob = champ_mdl.predict_proba(row_input)[0][1]

    return {
        'label'           : 'Needs Treatment' if pred == 1 else 'No Treatment',
        'probability'     : round(float(prob), 4),
        'needs_treatment' : bool(pred),
    }


# Demo
sample_input = {
    'Age': 28, 'Gender': 'Male', 'self_employed': 'No',
    'family_history': 'Yes', 'work_interfere': 'Often',
    'no_employees': '26-100', 'remote_work': 'No',
    'tech_company': 'Yes', 'benefits': 'Yes', 'care_options': 'Not sure',
    'wellness_program': 'No', 'seek_help': 'No', 'anonymity': 'Yes',
    'leave': 'Somewhat easy', 'mental_health_consequence': 'No',
    'phys_health_consequence': 'No', 'coworkers': 'Some of them',
    'supervisor': 'Yes', 'mental_health_interview': 'No',
    'phys_health_interview': 'Maybe', 'mental_vs_physical': 'Yes',
    'obs_consequence': 'No',
}
result = predict_single(sample_input)
print(f"\n[11] Demo Inference:")
print(f"     Result   : {result['label']}")
print(f"     Prob     : {result['probability']}")

print("\n" + "=" * 60)
print("  Pipeline selesai! Model siap digunakan.")
print("=" * 60)
