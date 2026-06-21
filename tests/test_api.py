"""API tests via FastAPI TestClient. Offline — uses bundled sample data, no LLM calls."""
import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


@pytest.fixture
def sample_mode(monkeypatch):
    monkeypatch.setenv("AINSTEIN_SAMPLE", "1")


@pytest.fixture
def no_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_health_ok():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "artifacts_available" in body
    assert "hf_token_configured" in body


def test_summary_available_in_sample_mode(sample_mode):
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["summary"]["num_papers"] == 6


def test_tiers_and_baselines_in_sample_mode(sample_mode):
    tiers = client.get("/api/tiers").json()["tiers"]
    assert {t["tier"] for t in tiers} == {"Oral", "Spotlight", "Poster"}
    baselines = client.get("/api/baselines").json()["baselines"]
    assert {b["method_name"] for b in baselines} == {
        "problem_restatement", "keyword_template", "abstract_copy",
    }


def test_papers_list_and_detail(sample_mode):
    listing = client.get("/api/papers?limit=10").json()
    assert listing["total"] == 6
    assert len(listing["items"]) == 6

    detail = client.get("/api/papers/iclr_0001")
    assert detail.status_code == 200
    assert "model_solution" in detail.json()


def test_paper_filter_by_tier(sample_mode):
    listing = client.get("/api/papers?tier=Oral").json()
    assert listing["total"] == 2
    assert all(p["tier"] == "Oral" for p in listing["items"])


def test_unknown_paper_returns_404(sample_mode):
    assert client.get("/api/papers/does_not_exist").status_code == 404


def test_demo_run_requires_token(no_token):
    resp = client.post("/api/demo/run", json={"row_index": 0})
    assert resp.status_code == 503
    assert "HUGGINGFACEHUB_API_TOKEN" in resp.json()["detail"]


def test_index_page_renders(sample_mode):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AINSTEIN" in resp.text


def test_paper_page_renders(sample_mode):
    resp = client.get("/paper/iclr_0001")
    assert resp.status_code == 200
    assert "Problem statement" in resp.text


def test_head_root_ok():
    resp = client.head("/")
    assert resp.status_code == 200
