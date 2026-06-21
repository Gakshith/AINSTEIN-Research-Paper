"""Tests for the AINSTEINController critique retry loop (offline, no LLM calls)."""
import main
from main import AINSTEINController


def _patch_stages(monkeypatch, controller, internal_pass, external_pass):
    """Replace _run_stage so no real pipeline/LLM runs, and stub critique statuses."""
    def fake_run_stage(stage_name, stage_obj):
        if stage_name == "Abstract Generalizer stage":
            return "PROBLEM_STATEMENT"
        if stage_name == "Solution stage":
            return "SOLUTION_CANDIDATE"
        if stage_name == "Data Validation stage":
            return True
        if stage_name == "Evaluation stage":
            return {"ok": True}
        return None

    # Loop logic under test must not depend on a real HuggingFace token.
    monkeypatch.setattr(main, "require_hf_token", lambda: "test-token")
    monkeypatch.setattr(controller, "_run_stage", fake_run_stage)
    statuses = {
        controller.internal_status_path: internal_pass,
        controller.external_status_path: external_pass,
    }
    monkeypatch.setattr(controller, "_read_status", lambda path: statuses.get(path, False))


def test_accepted_solution_is_returned_when_critique_passes(monkeypatch):
    controller = AINSTEINController(max_internal_attempts=1, max_external_attempts=1)
    _patch_stages(monkeypatch, controller, internal_pass=True, external_pass=True)

    p_final, z_final = controller.run()

    assert p_final == "PROBLEM_STATEMENT"
    assert z_final == "SOLUTION_CANDIDATE"
    assert controller.critique_passed is True


def test_best_effort_solution_kept_when_critique_never_passes(monkeypatch):
    controller = AINSTEINController(max_internal_attempts=1, max_external_attempts=1)
    _patch_stages(monkeypatch, controller, internal_pass=False, external_pass=False)

    p_final, z_final = controller.run()

    # Best-effort: the last generated candidate is kept, not silently discarded.
    assert z_final == "SOLUTION_CANDIDATE"
    assert controller.critique_passed is False
