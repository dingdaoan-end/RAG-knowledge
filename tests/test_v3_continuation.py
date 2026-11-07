import hashlib
import json

import pytest

from scripts.continue_v3_unstarted import preserve_and_replace, run_missing
from scripts.run_v3_comprehensive import InFlightError, ProgressStore


def test_continuation_only_calls_unstarted_units_never_failed_units(tmp_path):
    store = ProgressStore(tmp_path, "experiment")
    first, _ = store.begin("vector__qa_001", {})
    first.update(state="finished", status="failed")
    store.update(first)
    before = store.path("vector__qa_001").read_bytes()
    calls = []

    def execute(method, sample):
        calls.append(sample["id"])
        unit, _ = store.begin(method + "__" + sample["id"], {})
        unit.update(state="finished", status="success")
        store.update(unit)
        return unit

    units, created, stopped = run_missing([("vector", {"id": "qa_001"}), ("vector", {"id": "qa_002"})], store, execute)
    assert calls == ["qa_002"] and len(units) == 2 and len(created) == 1 and not stopped
    assert store.path("vector__qa_001").read_bytes() == before


def test_continuation_never_replays_inflight_and_stops_on_first_new_failure(tmp_path):
    store = ProgressStore(tmp_path, "experiment")
    store.begin("vector__qa_001", {})
    with pytest.raises(InFlightError):
        run_missing([("vector", {"id": "qa_001"})], store, lambda *args: pytest.fail("Must not call"))

    def fail(method, sample):
        return {"unit_id": method + "__" + sample["id"], "status": "failed", "http_attempts": []}
    units, created, stopped = run_missing([("bm25", {"id": "qa_002"}), ("bm25", {"id": "qa_003"})], store, fail)
    assert len(units) == len(created) == 1 and stopped


def test_active_summary_replacement_preserves_interrupted_bytes(tmp_path):
    path, snapshot = tmp_path / "active.json", tmp_path / "snapshot/old.json"
    original = b'{"partial":true}\r\n'
    path.write_bytes(original)
    preserve_and_replace(path, b'{"complete":true}\n', snapshot)
    assert snapshot.read_bytes() == original
    assert path.read_bytes() == b'{"complete":true}\n'
    assert not list(tmp_path.glob(".continued-*"))


def test_existing_hashed_snapshot_must_match_before_active_replacement(tmp_path):
    path, snapshot = tmp_path / "active.json", tmp_path / "snapshot/old.json"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"earlier snapshot")
    raw = b"active bytes"
    path.write_bytes(raw)
    hashed = snapshot.with_name(snapshot.stem + "." + hashlib.sha256(raw).hexdigest()[:16] + snapshot.suffix)
    hashed.write_bytes(b"partial snapshot")
    with pytest.raises(ValueError):
        preserve_and_replace(path, b"new", snapshot)
    assert path.read_bytes() == raw


def test_changed_prompt_blocks_continuation_before_config_or_paid_calls(tmp_path, monkeypatch):
    import scripts.continue_v3_unstarted as script
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "app/generation").mkdir(parents=True)
    (tmp_path / "scripts/run_v3_comprehensive.py").write_bytes(b"original runner")
    (tmp_path / "app/generation/prompt.py").write_bytes(b"changed prompt")
    manifest = {"experiment_id": "test", "config": {
        "runner_sha256": hashlib.sha256(b"original runner").hexdigest(),
        "prompt_source_sha256": hashlib.sha256(b"original prompt").hexdigest(),
    }}
    (tmp_path / "results/v3_experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results/v3_qa_comparison.json").write_text(json.dumps({"experiment_id": "test", "finished_units": 174}), encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Prompt"):
        script.main()
