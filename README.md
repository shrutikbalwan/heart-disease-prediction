# Disease Detection

## Disease validation study

Read [findings](DISEASE_VALIDATION_FINDINGS.md), the [detailed report](DISEASE_VALIDATION_REPORT.html),
and [calibration charts](outputs/disease_study/calibration_tradeoffs.html). The study
compares models, calibration, thresholds, age/code subgroups and blood-pressure
cleaning. It supports retaining the current model while documenting substantial
age-group differences and a source-data review need. The disease suite passes 22 tests.


## Advanced local application

Open this folder as its own workspace. The advanced application adds model
comparison, uncertainty/calibration, subgroup evaluation, single-record prediction,
CSV batches and downloadable results. No GitHub publication has been performed.

```powershell
python -m pip install -r requirements-dev.txt
python -m ml_projects.app --port 8503
```

Open http://127.0.0.1:8503 in your browser. On Windows, `start_app.cmd` is a
convenient launcher after dependencies are installed. Stop the server with Ctrl+C.
The app uses existing advanced artifacts; retrain explicitly when needed:

```powershell
python -m ml_projects.advanced --train
python -m ml_projects.advanced_deliver
python -m ml_projects.advanced_verify
python -m ml_projects.advanced --input outputs/advanced/example_inputs.csv --output verification/advanced_predictions.csv
```

Read `advanced_report.html`, `advanced_analysis.ipynb`, and `MODEL_CARD.md`.
The advanced notebook's five code cells were executed as Python; native IDE Run All
is not claimed for this new notebook. Restart the kernel when switching projects.
Original assignment artifacts remain available below.

| Metric | Historical-test value |
|---|---|
| accuracy | 0.7101 |
| precision | 0.6760 |
| recall | 0.8066 |
| F1 | 0.7355 |
| ROC_AUC | 0.7926 |
| average_precision | 0.7748 |
| Brier | 0.1846 |
| log_loss | 0.5504 |
| specificity | 0.6138 |

The original test set was inspected during earlier development. Results here are a historical benchmark, not new independent external validation.



This is an independent project. Open this folder in Antigravity or VS Code.
It includes its own dataset, Python package, trained model, notebook, report,
figures, dependency files and tests. You can copy this entire folder elsewhere.

## GitHub clone and CI

See [GitHub preparation and reproduction](GITHUB_PREPARATION.md) before using a
fresh clone. Generated artifacts and datasets are kept locally and excluded from
Git. The report links and app require regeneration after cloning. CI uses
synthetic inputs; full model verification requires the authorized dataset.

## Setup

Python 3.13 was used. Run commands from this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

Use `.venv\Scripts\python` instead of `python` below if using that environment.
Select its Python kernel, restart the notebook kernel, then Run All in
`analysis.ipynb`. Restarting is necessary if you previously used the combined
workspace kernel. `RETRAIN=False` checks existing trained results; set it to
`True` to retrain. Open `report.html` to read the report.

## Train, rebuild, predict and verify

```powershell
python -m ml_projects.train --project disease
python -m ml_projects.deliver --project disease
python -m ml_projects.predict --input outputs/example_inputs.csv --output outputs/predictions_demo.csv
python -m ml_projects.verify
```

Verification runs the local tests, recomputes held-out model metrics, executes
notebook Python cells and checks batch predictions against saved results.
It does not retrain or claim an interactive IDE run of this relocated copy.

The dataset is `datasets/project 3.xlsx`. Its source hash and tested package
versions are in `outputs/run.json`. Training uses a held-out test partition,
five-fold training-only cross-validation, and seed 42. Reports document model
limitations. Prediction examples are demonstrations, not independent validation.
