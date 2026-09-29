"""
Week 1 - Python for Machine Learning & Data Preprocessing

Takes the raw employee CSV and walks through, step by step:
  1. loading + inspecting
  2. duplicates
  3. quick EDA (plots saved to figures/)
  4. train/test split
  5. missing values
  6. categorical encoding
  7. scaling (standardization + min-max)
  8. feature selection
  9. a tiny model, only as a sanity check

Run:  python preprocess.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # so plots can be saved without a screen
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

TARGET = "Attrition"
NUM_COLS = ["Age", "Experience_Years", "Monthly_Income",
            "Satisfaction_Score", "Weekly_Work_Hours"]
CAT_COLS = ["Education", "Department", "City", "Job_Level"]


# small helper (basic Python: a function, a loop, a dict, an f-string)
def describe_missing(frame):
    """Print how many values are missing in each column that has any."""
    counts = frame.isnull().sum()
    for col, n in counts.items():
        if n > 0:
            pct = n / len(frame) * 100
            print(f"  {col:<20} {n:>3} missing ({pct:.1f}%)")


# ---------------------------------------------------------------
# 1. Load and inspect
# ---------------------------------------------------------------
print("=" * 60)
print("1. LOAD AND INSPECT")
print("=" * 60)
df = pd.read_csv("data/employee_data_raw.csv")
print("shape:", df.shape)
print(df.head().to_string())
print("\ndata types:")
print(df.dtypes.to_string())
print("\nmissing values:")
describe_missing(df)


# ---------------------------------------------------------------
# 2. Duplicates
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("2. DUPLICATES")
print("=" * 60)
n_dup = df.duplicated().sum()
print("duplicate rows found:", n_dup)
df = df.drop_duplicates().reset_index(drop=True)
print("shape after dropping them:", df.shape)


# ---------------------------------------------------------------
# 3. Exploratory data analysis
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("3. EDA")
print("=" * 60)
print(df[NUM_COLS].describe().round(2).to_string())

print("\nattrition split (0 = stayed, 1 = left):")
print(df[TARGET].value_counts().to_string())
print(df[TARGET].value_counts(normalize=True).round(3).to_string())

print("\nattrition rate by department:")
print(df.groupby("Department")[TARGET].mean().round(3).to_string())

# outlier check with the IQR rule (just looking, not removing anything)
print("\npossible outliers (IQR rule):")
for col in NUM_COLS:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"  {col:<20} {n_out} outside [{low:.1f}, {high:.1f}]")

# plot 1: histograms of numeric columns
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes.ravel(), NUM_COLS):
    ax.hist(df[col].dropna(), bins=20, color="#4C72B0", edgecolor="white")
    ax.set_title(col)
axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.savefig("figures/01_numeric_histograms.png", dpi=110)
plt.close()

# plot 2: missing values per column
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values()
plt.figure(figsize=(7, 3.5))
plt.barh(missing.index, missing.values, color="#DD8452")
plt.xlabel("number of missing values")
plt.title("Missing values per column (after removing duplicates)")
plt.tight_layout()
plt.savefig("figures/02_missing_values.png", dpi=110)
plt.close()

# plot 3: attrition rate by department
rate = df.groupby("Department")[TARGET].mean().sort_values()
plt.figure(figsize=(7, 3.5))
plt.barh(rate.index, rate.values, color="#55A868")
plt.xlabel("attrition rate")
plt.title("Attrition rate by department")
plt.tight_layout()
plt.savefig("figures/03_attrition_by_department.png", dpi=110)
plt.close()

# plot 4: correlation heatmap (numeric columns + target)
corr = df[NUM_COLS + [TARGET]].corr()
plt.figure(figsize=(6.5, 5.5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("figures/04_correlation_heatmap.png", dpi=110)
plt.close()
print("\nsaved 4 figures to figures/")


# ---------------------------------------------------------------
# 4. Train / test split
#    Done BEFORE filling/scaling so nothing from the test rows leaks in.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRAIN / TEST SPLIT")
print("=" * 60)
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("train:", X_train.shape, " test:", X_test.shape)


# ---------------------------------------------------------------
# 5. Missing values
#    numeric -> median of the TRAIN set, categorical -> most common
#    value of the TRAIN set. The same numbers are used on the test set.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("5. MISSING VALUES")
print("=" * 60)
X_train = X_train.copy()
X_test = X_test.copy()

medians = X_train[NUM_COLS].median()
modes = X_train[CAT_COLS].mode().iloc[0]
print("medians used:", medians.round(2).to_dict())
print("modes used:  ", modes.to_dict())

X_train[NUM_COLS] = X_train[NUM_COLS].fillna(medians)
X_test[NUM_COLS] = X_test[NUM_COLS].fillna(medians)
X_train[CAT_COLS] = X_train[CAT_COLS].fillna(modes)
X_test[CAT_COLS] = X_test[CAT_COLS].fillna(modes)

print("missing left in train:", int(X_train.isnull().sum().sum()),
      "| in test:", int(X_test.isnull().sum().sum()))


# ---------------------------------------------------------------
# 6. Categorical encoding (one-hot)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("6. ONE-HOT ENCODING")
print("=" * 60)
X_train = pd.get_dummies(X_train, columns=CAT_COLS, dtype=int)
X_test = pd.get_dummies(X_test, columns=CAT_COLS, dtype=int)
# make sure test has exactly the same columns as train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
print("columns before:", len(NUM_COLS) + len(CAT_COLS),
      "-> after:", X_train.shape[1])
print(list(X_train.columns))


# ---------------------------------------------------------------
# 7. Scaling
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("7. SCALING")
print("=" * 60)

# 7a. Min-max normalization, shown on one column so the effect is visible
mm = MinMaxScaler()
income_mm = mm.fit_transform(X_train[["Monthly_Income"]])
print("Monthly_Income before: min", X_train["Monthly_Income"].min(),
      "max", X_train["Monthly_Income"].max())
print("Monthly_Income after min-max: min", income_mm.min(),
      "max", income_mm.max())

# same thing done by hand with numpy, to check I understand the formula
inc = X_train["Monthly_Income"].to_numpy()
by_hand = (inc - inc.min()) / (inc.max() - inc.min())
print("hand-written formula matches sklearn:",
      np.allclose(by_hand, income_mm.ravel()))

# 7b. Standardization (used for the actual processed data)
scaler = StandardScaler()
X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])   # transform only!
print("\nafter standardization (train):")
print(X_train[NUM_COLS].agg(["mean", "std"]).round(3).to_string())


# ---------------------------------------------------------------
# 8. Feature selection
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("8. FEATURE SELECTION (SelectKBest, ANOVA F-test)")
print("=" * 60)
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X_train, y_train)

scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
print("F-scores, highest first:")
print(scores.round(2).to_string())

selected = X_train.columns[selector.get_support()].tolist()
print("\nkept 10 features:", selected)

plt.figure(figsize=(7, 5))
scores.sort_values().plot(kind="barh", color="#8172B2")
plt.xlabel("F-score")
plt.title("Feature scores (SelectKBest)")
plt.tight_layout()
plt.savefig("figures/05_feature_scores.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 9. Save processed data
# ---------------------------------------------------------------
train_out = X_train.copy()
train_out[TARGET] = y_train.values
test_out = X_test.copy()
test_out[TARGET] = y_test.values
train_out.to_csv("data/processed_train.csv", index=False)
test_out.to_csv("data/processed_test.csv", index=False)
pd.Series(selected, name="selected_feature").to_csv(
    "data/selected_features.csv", index=False)
print("\nsaved data/processed_train.csv, processed_test.csv, selected_features.csv")


# ---------------------------------------------------------------
# 10. Sanity check with a small model
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("10. SANITY CHECK MODEL")
print("=" * 60)
model = LogisticRegression(max_iter=1000)
model.fit(X_train[selected], y_train)
acc = accuracy_score(y_test, model.predict(X_test[selected]))
baseline = max(y_test.mean(), 1 - y_test.mean())   # always guess the majority class
print(f"logistic regression accuracy: {acc:.3f}")
print(f"always-predict-'stayed' baseline: {baseline:.3f}")
print("The model runs on the processed data, which is all this check is for.")
print("The synthetic data has only weak patterns, so accuracy is modest.")
