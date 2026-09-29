# Python for ML – Data Preprocessing (Week 1)

Week 1 internship task: learn the basics of Python, NumPy and Pandas, and clean/preprocess a sample dataset while documenting each step.

## Dataset
`data/employee_data_raw.csv` – a **synthetic** employee dataset I generated for this task (it is not real company data). 508 rows, 10 columns:

- numeric: Age, Experience_Years, Monthly_Income, Satisfaction_Score, Weekly_Work_Hours
- categorical: Education, Department, City, Job_Level
- target: Attrition (1 = left, 0 = stayed)

It has some deliberate problems to practise on: 8 duplicate rows and 64 missing values across 5 columns.

## What the code does
1. Load the CSV and inspect shape, types, missing values
2. Remove duplicate rows (508 → 500)
3. EDA: summary stats, class balance, attrition by department, IQR outlier check, 4 plots
4. Train/test split (80/20, stratified) *before* any filling or scaling
5. Missing values: median (numeric) / most frequent (categorical), computed on train only
6. One-hot encoding (9 columns → 23)
7. Scaling: min-max demo on one column, StandardScaler on the numeric features
8. Feature selection with SelectKBest (ANOVA F-test), top 10 kept
9. Save processed data
10. Small logistic regression as a sanity check only

## Run it
```bash
pip install -r requirements.txt
python preprocess.py
```
or open `week1_preprocessing.ipynb`.

## Files
| file | what it is |
|---|---|
| `preprocess.py` | full workflow as one script |
| `week1_preprocessing.ipynb` | same workflow in notebook form, with outputs |
| `data/employee_data_raw.csv` | raw data |
| `data/processed_train.csv`, `processed_test.csv` | encoded + scaled data (all 23 features + target) |
| `data/selected_features.csv` | the 10 features picked by SelectKBest |
| `figures/` | EDA and feature-score plots |

## Result / caveat
The logistic regression gets 0.64 accuracy on the test set, which is slightly *below* the 0.67 you'd get by always predicting "stayed". That's fine for this task: the data is synthetic with weak patterns, and the model is only there to show the processed data works end to end. The point of the week is the preprocessing, not the model.