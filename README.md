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

- [main.py](main.py): single-paper pipeline
- [evaluation_result.py](evaluation_result.py): full dataset batch evaluation
- [app.py](app.py): web dashboard + API (FastAPI)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + tests and optional extras
```

## Environment

The LLM stages call HuggingFace inference endpoints (`deepseek-ai/DeepSeek-R1`) and need an
API token. Copy `.env.example` to `.env` and set it (or export it):

```bash
export HUGGINGFACEHUB_API_TOKEN=hf_xxx   # or HF_TOKEN
```

Without a token the pipeline fails fast with a clear message, and the web app's live demo is
disabled (the results dashboard still works).

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

Configure the selected paper in [config/config.yaml](config/config.yaml) using:
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

Batch evaluation writes results to [artifacts/evaluation](artifacts/evaluation):

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

## Web app (dashboard + live demo)

A FastAPI app serves a results dashboard (charts, per-paper drill-down, baselines) and a live
single-paper demo.

```bash
uvicorn app:app --reload
# open http://127.0.0.1:8000   (interactive API docs at /docs)
```

- The dashboard reads existing artifacts from `artifacts/evaluation/`. Set `AINSTEIN_SAMPLE=1`
  to preview with bundled sample data before you've run a real evaluation.
- The live demo (`/demo`) runs the full pipeline on one paper; it requires
  `HUGGINGFACEHUB_API_TOKEN` and is disabled with a clear message otherwise.

## Docker

Build:

```bash
docker build -t ainstein-eval .
```

Serve the web app (default):

```bash
docker run --rm -p 8000:8000 -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN \
  -v "$(pwd)/artifacts:/app/artifacts" ainstein-eval
```

Run batch evaluation:

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" ainstein-eval python evaluation_result.py
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
