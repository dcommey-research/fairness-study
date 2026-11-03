# Group sparsity and fairness–performance tradeoffs

This repository contains the code and the results for the article *Group Sparsity and Fairness–Performance Tradeoffs: A Controlled Evaluation of Bias Mitigation on Seven Tabular Datasets* by R. A. Sarpong and D. Commey.

The study compares three bias mitigation methods with matched baselines. The methods are reweighing (RW), threshold optimization (TO) and exponentiated-gradient reduction (EG). The study uses four learners and seven public datasets. Each comparison uses five repetitions of five-fold cross-validation.

The version for the article has the tag `sncs-revision-1`.

## Contents

| Path | Contents |
|---|---|
| `revision/data.py` | Dataset download, filters, outcome coding and sensitive groups |
| `revision/run.py`, `revision/evaluation.py` | Main experiment and metrics |
| `revision/PROTOCOL.md` | Fixed protocol of the experiment |
| `revision/extended/compas_encoding.py` | COMPAS race-grouping ablation |
| `revision/report.py` | Numerical result files from the saved evaluations |
| `revision/extended/publication.py` | Intervals, noise floor, figures and tables |
| `revision/verify.py`, `revision/extended/verify_compas.py` | Independent checks from the saved prediction arrays |
| `revision/generated/` | Numerical results, figures and tables in the article |
| `tests/` | Unit tests |

The release `sncs-revision-1` has two more files:

| File | Contents |
|---|---|
| `Online_Resource_1.zip` | Online Resource 1 of the article: the numerical results as CSV files, with a README |
| `fold_predictions.zip` | Saved predictions, split indices and records for each evaluation (`results/` folder) |

## Requirements

- Python 3.12. The recorded run used Python 3.12.14 on macOS (Apple silicon).
- The packages in `requirements-revision.lock.txt`.

## Check the released results

Use this procedure to check the results without a new run.

1. Make a virtual environment:

   ```sh
   python3.12 -m venv .venv
   ```

2. Install the packages:

   ```sh
   .venv/bin/python -m pip install -r requirements-revision.lock.txt pytest
   ```

3. Run the unit tests:

   ```sh
   .venv/bin/python -m pytest
   ```

4. Download `fold_predictions.zip` from the release `sncs-revision-1`.
5. Extract `fold_predictions.zip` in the repository folder. The files go into `results/`.
6. Do the checks of the main experiment:

   ```sh
   .venv/bin/python -m revision.verify --input results/revision_verified
   ```

7. Do the checks of the COMPAS ablation:

   ```sh
   .venv/bin/python -m revision.extended.verify_compas
   ```

Each check writes its status to a JSON file in `revision/generated/`.

## Run the full study

The full run takes many hours. ACS Income uses most of the time.

1. Do steps 1 to 3 of the procedure above.
2. Download the ACS data:

   ```sh
   .venv/bin/python -m revision.fetch_data
   ```

   The other datasets download when the run starts.

3. Run all steps:

   ```sh
   PYTHON_BIN=.venv/bin/python ./run_study.sh
   ```

A run does not continue into an output folder if the code hash or the arguments are different. The report step stops if the ledger is not complete. The run records each failed evaluation. A failure occurs when a group in a TO calibration set does not have one of the two outcome classes.

To do a short test run, use this command. Its results are not in the article.

```sh
.venv/bin/python main.py --datasets german --repeats 1 --folds 2 --no-ablation --bootstrap 50 --output results/smoke
```

## Data

All datasets are public:

- Adult: OpenML.
- German Credit, Heart Disease (Cleveland) and Bank Marketing: UCI Machine Learning Repository.
- COMPAS: ProPublica.
- LSAC: the raw SAS file. The code keeps Black and White students only, as AIF360 does.
- ACS Income: California 2018 person PUMS (`data/folktables/2018/1-Year/psam_p06.csv`).

The filters and the outcome coding are in `revision/data.py`. The repository does not contain the downloaded data.

## License and citation

The license is the MIT License. To cite this software, use the data in `CITATION.cff`.
