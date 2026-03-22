# AINSTEIN-Research-Paper

Implementation of a research-engineering pipeline inspired by **AINSTEIN: Assessing the Feasibility of AI-Generated Approaches to Research Problems**.

## What This Project Does

The pipeline:
- ingests paper metadata into a dataset
- selects a paper from the dataset
- downloads the paper PDF using `pdf_url`
- extracts the paper abstract and a paper-derived reference solution
- generalizes the abstract into a method-agnostic problem statement
- generates an AI solution
- critiques the solution internally and externally
- evaluates the generated solution against the reference paper solution

## Main Entry Points

- [main.py](/Users/gojuruakshith/PycharmProjects/AINSTEIN-Research-Paper/main.py): single-paper pipeline
- [evaluation_result.py](/Users/gojuruakshith/PycharmProjects/AINSTEIN-Research-Paper/evaluation_result.py): full dataset batch evaluation

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

`artifacts/data_ingestion/data.csv` should contain:

```csv
paper_id,title,abstract,pdf_url,tier
```

Expected `tier` values:
- `Oral`
- `Spotlight`
- `Poster`

## How To Run

### Single-paper run

Configure the selected paper in [config/config.yaml](/Users/gojuruakshith/PycharmProjects/AINSTEIN-Research-Paper/config/config.yaml) using:
- `generalizer.paper_id`
- or `generalizer.row_index`

Then run:

```bash
./venv/bin/python main.py
```

### Full dataset batch run

```bash
./venv/bin/python evaluation_result.py
```

## Output Files

Batch evaluation writes results to [artifacts/evaluation](/Users/gojuruakshith/PycharmProjects/AINSTEIN-Research-Paper/artifacts/evaluation):

- `evaluation_results.csv`
- `baseline_results.csv`
- `evaluation_summary.csv`
- `evaluation_tier_summary.csv`
- `baseline_summary.csv`
- `overall_results_table.md`
- `tier_results_table.md`
- `baseline_results_table.md`
- `experiment_report.md`
- `plots/`

## Docker

Build:

```bash
docker build -t ainstein-eval .
```

Run batch evaluation:

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" ainstein-eval
```

Run single-paper mode:

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" ainstein-eval python main.py
```

## Tests

```bash
./venv/bin/python -m pytest tests
```

## Citation

Paper page:
- https://openreview.net/pdf/32789ed9a5cb7c59bb0d0ccd108f5ce441e6c38e.pdf

Recommended citation entry:

```bibtex
@article{ainstein2025,
  title={AINSTEIN: Assessing the Feasibility of AI-Generated Approaches to Research Problems},
  author={Shambhavi Mishra,Gaurav Sahu,Marco Pedersoli1, Laurent Charlin, Jose Dolz,Christopher Pal2},
  journal={ICLR 2026 submission},
  year={2025},
  url={https://openreview.net/pdf/32789ed9a5cb7c59bb0d0ccd108f5ce441e6c38e.pdf}
}
```

Note:
- the OpenReview submission currently shows **Anonymous Authors** because it is under double-blind review.
