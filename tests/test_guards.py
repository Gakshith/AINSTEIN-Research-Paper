from pathlib import Path

import pytest

from src.AINSTEIN.utils.common import require_hf_token, hf_token_available
from src.AINSTEIN.components.data_validation import DataValidation
from src.AINSTEIN.entity.config_entity import DataValidationConfig


def _clear_hf_env(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_require_hf_token_raises_clear_error_when_absent(monkeypatch):
    _clear_hf_env(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        require_hf_token()
    assert "HUGGINGFACEHUB_API_TOKEN" in str(exc.value)


def test_require_hf_token_returns_token_when_present(monkeypatch):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_test123")
    assert require_hf_token() == "hf_test123"


def test_require_hf_token_accepts_hf_token_alias(monkeypatch):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_alias")
    assert require_hf_token() == "hf_alias"


def test_hf_token_available_reflects_env(monkeypatch):
    _clear_hf_env(monkeypatch)
    assert hf_token_available() is False
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_x")
    assert hf_token_available() is True


def test_data_validation_missing_file_raises_clear_error(tmp_path: Path):
    config = DataValidationConfig(
        root_dir=tmp_path,
        data_path=tmp_path / "does_not_exist.csv",
        STATUS_FILE=tmp_path / "status.txt",
        all_schema={"paper_id": "str"},
    )
    with pytest.raises(FileNotFoundError) as exc:
        DataValidation(config).validate_dataset()
    assert "data ingestion" in str(exc.value).lower()
