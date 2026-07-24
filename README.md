# OOI RCA Copilot

A domain-grounded LLM harness for accessing and analyzing **Ocean Observatories
Initiative (OOI) Regional Cabled Array (RCA)** data through natural language.

The LLM is constrained to a small set of structured tools — it never executes
code directly. It searches curated RCA instrumentation metadata to identify
valid reference designators, methods, and streams, builds data requests the user
explicitly approves, and processes results through auditable xarray pipelines.

## Data backends

- **M2M** — asynchronous requests to the OOI Machine-to-Machine API (full
  catalog coverage; request → THREDDS poll → NetCDF download).
- **Cloud Zarr (fast path)** — direct, lazy reads from the public `ooi-data` S3
  Zarr store when a stream is published there; returns in seconds with no
  THREDDS wait. Suggested automatically when the fast path is available.

## Setup

Requires Python **3.11+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .          # runtime deps + editable install
pip install -e ".[dev]"   # also install ruff / pytest / flake8
```

Point your editor's interpreter at `.venv` so imports resolve.

### Secrets

Create `.streamlit/secrets.toml` (git-ignored) with at least:

```toml
FIREWORKS_API_KEY = "..."   # cloud LLM provider
OOI_USERNAME = "..."        # OOI M2M credentials
OOI_TOKEN = "..."
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## Tests

```bash
pytest
```
