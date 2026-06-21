"""AINSTEIN web app: FastAPI API + server-rendered dashboard.

Run locally:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000

The dashboard visualizes existing evaluation artifacts and offers a live
single-paper demo (enabled when HUGGINGFACEHUB_API_TOKEN is set).
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web import runner, services

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AINSTEIN Dashboard API",
    description="Visualize AI-generated research-solution evaluations and run a live demo.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ----------------------------------------------------------------------------- API
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "artifacts_available": services.artifacts_available(),
        "hf_token_configured": runner.token_configured(),
    }


@app.get("/api/summary")
def summary() -> dict:
    data = services.get_summary()
    return {"available": data is not None, "summary": data}


@app.get("/api/tiers")
def tiers() -> dict:
    return {"tiers": services.get_tiers()}


@app.get("/api/baselines")
def baselines() -> dict:
    return {"baselines": services.get_baselines()}


@app.get("/api/papers")
def papers(tier: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    return services.get_papers(tier=tier, limit=limit, offset=offset)


@app.get("/api/papers/{paper_id}")
def paper(paper_id: str) -> dict:
    data = services.get_paper(paper_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Paper not found in results")
    return data


@app.get("/api/report")
def report() -> dict:
    text = services.get_report()
    return {"available": text is not None, "report": text}


@app.get("/api/plots/{name}")
def plot(name: str):
    path = services.plot_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/demo/run")
def demo_run(payload: dict | None = None):
    payload = payload or {}
    row_index = payload.get("row_index")
    paper_id = payload.get("paper_id")
    try:
        job = runner.start_run(row_index=row_index, paper_id=paper_id)
    except RuntimeError as exc:
        reason = str(exc)
        if reason == "no_token":
            return JSONResponse(
                status_code=503,
                content={"detail": "HUGGINGFACEHUB_API_TOKEN is not set; live demo is disabled."},
            )
        if reason == "busy":
            return JSONResponse(
                status_code=409,
                content={"detail": "A demo run is already in progress. Try again shortly."},
            )
        raise
    return job


@app.get("/api/demo/{job_id}")
def demo_status(job_id: str) -> dict:
    job = runner.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# --------------------------------------------------------------------------- Pages
@app.get("/", response_class=HTMLResponse)
@app.head("/")
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"available": services.artifacts_available()}
    )


@app.get("/paper/{paper_id}", response_class=HTMLResponse)
def paper_page(request: Request, paper_id: str):
    data = services.get_paper(paper_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return templates.TemplateResponse(request, "paper.html", {"paper": data})


@app.get("/demo", response_class=HTMLResponse)
def demo_page(request: Request):
    return templates.TemplateResponse(
        request, "demo.html", {"hf_token_configured": runner.token_configured()}
    )
