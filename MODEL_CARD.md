# Advanced model card

## Intended use

Local educational exploration, reproducible evaluation and batch inference.

## Training and selection

Candidate families: linear model and histogram gradient boosting. Preprocessing
is fitted inside each three-fold search. Selection uses neg_log_loss.
The selected model is hist_gradient_boosting. Configuration is in config.json.
Partition row IDs are in outputs/advanced/partitions.json. Calibration and
threshold selection use separate rows from model fitting. The original assignment
models and their reports remain in outputs/ and analysis.ipynb.

## Evaluation history

The original test set was inspected during earlier development. Results here are a historical benchmark, not new independent external validation.
No new external data was acquired. Bootstrap intervals resample rows and do not
account for selection uncertainty or geographic dependence. Subgroup comparisons
are descriptive; small samples and single-class groups are flagged.

## Limitations

Educational research only, not clinical diagnosis or future-onset prediction. No external clinical validation. Age is in days; the height unit is unresolved, so BMI is omitted. Gender codes are not assigned invented meanings. Invalid blood-pressure combinations are imputed under the documented data-quality rule.

## Operational scope

The app binds to loopback only, stores no submitted records, limits batch size,
and validates numeric inputs. Range warnings are heuristics, not a fitted drift
detector. Joblib models must only be loaded from trusted sources. This local demo
does not include authentication, production hosting or clinical approval.

## Evidence

- outputs/advanced/summary.json: metrics, source/model hashes and configuration
- outputs/advanced/calibration_validation.json: calibration comparison (classification)
- advanced_report.html: readable model evidence
- verification/advanced_checks.json: final automated checks

## Nested validation follow-up

Five outer folds and three inner folds on original training rows support cleaned
boosting over cleaned logistic regression. At the validation-selected 80% sensitivity
policy, mean outer sensitivity is 80.4% and specificity 62.6%. Sensitivity differs
substantially by age: 58.5% below 40 versus 91.1% at 55-69; specificity is 92.8% versus
28.6%. No age-70-plus training records were available. Calibration changes are tiny.
Raw-BP boosting scored somewhat better and triggered source-data review, not a
policy change. Model weights, calibration and cleaning remain unchanged. See
DISEASE_VALIDATION_FINDINGS.md for counts, uncertainty and interpretation limits.
