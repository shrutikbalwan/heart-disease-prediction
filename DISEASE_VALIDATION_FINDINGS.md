# Disease validation study: findings

**Decision: keep the current cleaned gradient-boosting application model.**
The study supports that model family over cleaned logistic regression, while exposing
age-group disparities and a blood-pressure data-quality issue requiring further review.
No model weights, thresholds, calibration artifacts or cleaning policies were changed.
This is an educational project without clinical or external validation.

## Study design

The original 56,000 training records were evaluated in five stratified outer folds,
with three inner folds for hyperparameter selection. All 14,000 original historical-test
records were excluded. Each outer development partition was split into 60% fitting,
20% calibration and 20% threshold-validation rows. All four variants used the same
partitions. Imputation, scaling and encoding were fitted inside each inner search.

Variants were logistic regression and histogram gradient boosting, each with and
without the existing blood-pressure cleaning rule. Inner selection used log loss.
Sigmoid calibration used separate rows and a frozen fitted base estimator. Thresholds
for 70%, 80% and 90% validation sensitivity were then selected on another partition.
The primary comparison uses the 80% target. No rows were dropped.

## Mean outer-fold results

All probability scores below use calibrated predictions; lower log loss and Brier
score are better. These are internal CV results, not historical-test or clinical results.

| Variant | ROC-AUC | Log loss | Brier | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Boosting, cleaned BP | 0.7968 | 0.5461 | 0.1827 | 80.4% | 62.6% | 0.7382 |
| Boosting, raw BP | 0.8021 | 0.5401 | 0.1800 | 80.6% | 63.8% | 0.7435 |
| Logistic, cleaned BP | 0.7890 | 0.5631 | 0.1881 | 80.4% | 61.4% | 0.7341 |
| Logistic, raw BP | 0.7778 | 0.5829 | 0.1944 | 80.1% | 59.4% | 0.7257 |

Cleaned boosting beat cleaned logistic regression on log loss in all five folds.
Its mean log-loss advantage was 0.01691. That supports retaining the existing model
family; it does not establish clinical validity or independently evaluate the saved
application model, since each fold model was freshly fitted.

## Calibration findings

For cleaned boosting, mean log loss changed only from **0.546162 to 0.546144** after
sigmoid calibration. Brier score changed from **0.182729 to 0.182723**. Those changes
are tiny. Calibration slightly worsened the aggregate scores for cleaned logistic
and raw-BP boosting. We therefore do not claim that calibration always improves
performance or that the resulting probabilities are perfectly calibrated.

The [offline calibration and trade-off charts](outputs/disease_study/calibration_tradeoffs.html)
show reliability curves and threshold policies. Fixed-decile reliability tables
include bin counts and approximate Wilson intervals for observed event frequency.
Plotly is bundled; viewing the chart requires no internet. Its fixed width supports
horizontal scrolling in narrow windows.

## Sensitivity-specificity trade-offs

Cleaned boosting, mean outer-fold performance:

| Validation policy | Evaluation sensitivity | Evaluation specificity | F1 |
|---|---:|---:|---:|
| Fixed threshold 0.5 | 69.1% | 76.7% | 0.7184 |
| Target 70% sensitivity | 70.5% | 75.4% | 0.7223 |
| Target 80% sensitivity | 80.4% | 62.6% | 0.7382 |
| Target 90% sensitivity | 90.2% | 42.4% | 0.7276 |

Increasing sensitivity produces more false positives. Validation targets are neither
clinical recommendations nor guarantees on other data. Thresholds vary by fold;
these policy comparisons do not prescribe a universal operating threshold.

## Age and gender-code differences

Cleaned boosting at each fold's 80%-target threshold:

| Age band | Records | Positive labels | Sensitivity | Approximate 95% interval | Specificity |
|---|---:|---:|---:|---:|---:|
| Under 40 | 1,478 | 349 | 58.5% | 53.2%-63.5% | 92.8% |
| 40-54 | 29,599 | 12,403 | 67.8% | 67.0%-68.6% | 79.8% |
| 55-69 | 24,923 | 15,231 | 91.1% | 90.7%-91.6% | 28.6% |

Overall sensitivity hides substantial subgroup differences. The younger band misses
many positive records, while the oldest represented band has many false positives.
No training observations were available at age 70 or above, so that band's performance
is unmeasured. Age is converted from source days using 365.25.

Dataset gender code 1.0 has 36,450 records, sensitivity 80.8% and specificity 63.0%.
Code 2.0 has 19,550 records, sensitivity 79.6% and specificity 62.0%. The original code
values are preserved without assigning invented meanings. Full subgroup and age/code
intersection tables include counts, confusion matrices, probability scores and
uncertainty bounds. These differences are descriptive, not causal or proof of fairness.

Wilson intervals use an independent-binomial approximation. They do not capture
model-selection uncertainty, dependence between observations or changes in deployment
populations. Small samples are flagged if there are fewer than 100 records or fewer
than 30 positives or negatives.

## Blood-pressure cleaning sensitivity

1,033 original training rows violated the existing broad data-quality rule. The clean
policy replaces both BP fields with missing values, then uses fitting-row imputation.
The raw policy retains finite numeric anomalies solely as a sensitivity experiment.
These bounds are data-quality assumptions, not diagnostic criteria.

Raw-BP boosting improved mean log loss by 0.00606 and AUC by 0.00526 relative to cleaned
boosting. This crossed the predefined threshold for **further review**, not deployment.
In contrast, retaining raw BP values worsened logistic-regression performance. The
cleaning effect depends on the model family.

785 of the 1,033 anomalous records had positive labels. For cleaned boosting, sensitivity
on these records was 63.2%, versus 80.9% on records within the rule. This association
could reflect useful source information or data-collection artifacts; the experiment
does not establish which. The correct next evidence is an audit of source units,
transcription and collection processes, rather than silently accepting invalid values.
The existing cleaning policy remains unchanged.

## Verification and reproducibility

**22 disease tests passed**, including nine study-specific tests. They check BP-cleaning
behavior, row/group counts, historical-test exclusion, split disjointness, complete
out-of-fold coverage, threshold selection from validation rows, probability/classification
metric recomputation, calibration bins, subgroup uncertainty and unchanged model hashes.
A second full study reproduced identical mean metrics and decisions. Charts were
visually reviewed in Chrome; the application explanation now mentions age-group limits.

The audit caught default CSV parsing rounding some probabilities extremely close to 1,
which slightly changed recomputed log loss. Use round-trip parsing for exact reproduction:

```python
predictions = pandas.read_csv(
    'outputs/disease_study/out_of_fold_predictions.csv',
    float_precision='round_trip',
)
```

Reproduce from this project folder:

```powershell
python -m pip install -r requirements-study.txt
python -m ml_projects.disease_study
python -m ml_projects.advanced_verify
```

Evidence: verification/disease_study_checks.json and verification/advanced_pytest.txt.
The [detailed report](DISEASE_VALIDATION_REPORT.html) and outputs/disease_study/ contain
all partitions, inner searches, predictions, threshold policies and subgroup tables.
Verification used installed laptop dependencies; a clean installation remains unverified.

## Remaining limits

The dataset was studied previously. Internal nested CV does not create new external,
temporal or clinical validation. Height units remain unresolved and BMI is omitted.
No subgroup-specific threshold was introduced from these evaluation results. Any future
model or policy change needs a new documented experiment and ideally new source data.
Nothing was published or pushed to GitHub.

Method reference: [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration).
