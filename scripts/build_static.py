#!/usr/bin/env python3
"""Build a static export of the AINSTEIN dashboard for GitHub Pages.

There is no backend at runtime on Pages: the dashboard reads pre-generated
``api/*.json`` files, and every paper page is pre-rendered to flat HTML. Paths
are rewritten to be relative so the site works under a project subpath
(``/<repo>/``).

Run from the repo root:

    AINSTEIN_SAMPLE=1 python scripts/build_static.py

Output goes to ``site/`` (gitignored; rebuilt by CI).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Populate the dashboard from bundled sample data; force the demo into its
# static-preview state regardless of any local token.
os.environ.setdefault("AINSTEIN_SAMPLE", "1")
os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
os.environ.pop("HF_TOKEN", None)

from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from web import services  # noqa: E402

OUT = ROOT / "site"
env = Environment(
    loader=FileSystemLoader(str(ROOT / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)


def staticize(html: str) -> str:
    """Rewrite server-absolute paths to relative ones and flag static mode."""
    html = html.replace("/static/", "static/")
    html = html.replace('href="/demo"', 'href="demo.html"')
    html = html.replace('href="/"', 'href="index.html"')
    html = html.replace(
        '<script src="static/js/app.js',
        '<script>window.AINSTEIN_STATIC=true;</script>\n  <script src="static/js/app.js',
    )
    return html


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", str(pid))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    available = services.artifacts_available()

    write(OUT / "index.html",
          staticize(env.get_template("index.html").render(available=available)))
    write(OUT / "demo.html",
          staticize(env.get_template("demo.html").render(
              static_preview=True, hf_token_configured=False)))

    papers = services.get_papers(limit=1_000_000)
    for item in papers["items"]:
        pid = item["paper_id"]
        paper = services.get_paper(pid)
        if not paper:
            continue
        write(OUT / f"paper-{safe(pid)}.html",
              staticize(env.get_template("paper.html").render(paper=paper)))

    api = OUT / "api"
    write(api / "summary.json",
          json.dumps({"available": available, "summary": services.get_summary()}))
    write(api / "tiers.json", json.dumps({"tiers": services.get_tiers()}))
    write(api / "baselines.json", json.dumps({"baselines": services.get_baselines()}))
    write(api / "papers.json", json.dumps(services.get_papers(limit=1_000_000)))

    shutil.copytree(ROOT / "static", OUT / "static")
    (OUT / ".nojekyll").write_text("")

    print(f"Built static site -> {OUT}  ({len(papers['items'])} paper pages)")


if __name__ == "__main__":
    main()
