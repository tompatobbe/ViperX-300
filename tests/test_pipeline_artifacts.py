"""
tests/test_pipeline_artifacts.py
Run with:  pytest tests/test_pipeline_artifacts.py -v
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable when running from any CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))
import pipeline_artifacts as pa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_csv(tmp_path: Path) -> Path:
    """A minimal CSV file that exists on disk (checksum can be computed)."""
    p = tmp_path / "run1.csv"
    p.write_text("time,waist_pos\n0.0,0.0\n0.02,0.01\n")
    return p


@pytest.fixture
def phi() -> np.ndarray:
    return np.random.default_rng(42).standard_normal(78)


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

class TestConfigHash:

    def test_deterministic(self):
        cfg = {"fs": 50.0, "fc": 10.0, "stride": 1}
        assert pa.config_hash(cfg) == pa.config_hash(cfg)

    def test_key_order_invariant(self):
        assert pa.config_hash({"a": 1, "b": 2}) == pa.config_hash({"b": 2, "a": 1})

    def test_different_values_differ(self):
        assert pa.config_hash({"fs": 50.0}) != pa.config_hash({"fs": 100.0})

    def test_different_keys_differ(self):
        assert pa.config_hash({"fs": 50.0}) != pa.config_hash({"fc": 50.0})

    def test_default_length_is_8(self):
        assert len(pa.config_hash({})) == 8

    def test_custom_length(self):
        assert len(pa.config_hash({}, length=12)) == 12

    def test_empty_config_stable(self):
        # An empty config must produce a consistent, non-empty hash.
        h = pa.config_hash({})
        assert isinstance(h, str) and len(h) == 8

    def test_numpy_scalar_serialisable(self):
        cfg = {"n": np.int64(5), "v": np.float32(1.0)}
        h = pa.config_hash(cfg)
        assert len(h) == 8


# ---------------------------------------------------------------------------
# artifact_path
# ---------------------------------------------------------------------------

class TestArtifactPath:

    def test_returns_npy_and_json(self, tmp_path, dummy_csv):
        npy, js = pa.artifact_path(dummy_csv, "sysid", "1.0", {}, tmp_path)
        assert npy.suffix == ".npy"
        assert js.suffix == ".json"

    def test_subdir_is_type_name(self, tmp_path, dummy_csv):
        npy, _  = pa.artifact_path(dummy_csv, "sysid", "1.0", {}, tmp_path, suffix=".npy")
        urdf, _ = pa.artifact_path(dummy_csv, "sysid", "1.0", {}, tmp_path, suffix=".urdf")
        assert npy.parent  == tmp_path / "npy"
        assert urdf.parent == tmp_path / "urdf"

    def test_pipeline_name_in_filename(self, tmp_path, dummy_csv):
        npy, _ = pa.artifact_path(dummy_csv, "mypipe", "1.0", {}, tmp_path)
        assert "mypipe" in npy.name

    def test_version_slug_in_filename(self, tmp_path, dummy_csv):
        npy, _ = pa.artifact_path(dummy_csv, "p", "2.3", {}, tmp_path)
        assert "v2-3" in npy.name     # dots replaced with dashes

    def test_cfg_hash_in_filename(self, tmp_path, dummy_csv):
        cfg = {"fs": 50.0}
        npy, _ = pa.artifact_path(dummy_csv, "p", "1.0", cfg, tmp_path)
        assert f"cfg-{pa.config_hash(cfg)}" in npy.name

    def test_fully_deterministic(self, tmp_path, dummy_csv):
        args = (dummy_csv, "sysid_feasible", "1.1", {"fs": 50.0, "stride": 1})
        assert (pa.artifact_path(*args, tmp_path)
                == pa.artifact_path(*args, tmp_path))

    def test_version_change_changes_path(self, tmp_path, dummy_csv):
        p1, _ = pa.artifact_path(dummy_csv, "p", "1.0", {}, tmp_path)
        p2, _ = pa.artifact_path(dummy_csv, "p", "1.1", {}, tmp_path)
        assert p1 != p2

    def test_config_change_changes_path(self, tmp_path, dummy_csv):
        p1, _ = pa.artifact_path(dummy_csv, "p", "1.0", {"fs": 50.0}, tmp_path)
        p2, _ = pa.artifact_path(dummy_csv, "p", "1.0", {"fs": 100.0}, tmp_path)
        assert p1 != p2

    def test_different_csv_stems_differ(self, tmp_path):
        csv_a = tmp_path / "run_a.csv"; csv_a.touch()
        csv_b = tmp_path / "run_b.csv"; csv_b.touch()
        p1, _ = pa.artifact_path(csv_a, "p", "1.0", {}, tmp_path)
        p2, _ = pa.artifact_path(csv_b, "p", "1.0", {}, tmp_path)
        assert p1 != p2


# ---------------------------------------------------------------------------
# save_artifact / load_artifact
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_creates_npy_and_json(self, tmp_path, dummy_csv, phi):
        npy_p, json_p = pa.save_artifact(
            phi, dummy_csv, "test_pipe", "1.0", {"fs": 50.0}, tmp_path
        )
        assert npy_p.exists()
        assert json_p.exists()

    def test_roundtrip_values(self, tmp_path, dummy_csv, phi):
        npy_p, _ = pa.save_artifact(
            phi, dummy_csv, "test_pipe", "1.0", {}, tmp_path
        )
        assert np.allclose(phi, np.load(npy_p))

    def test_sidecar_fields(self, tmp_path, dummy_csv, phi):
        cfg = {"fs": 50.0, "stride": 2}
        _, json_p = pa.save_artifact(phi, dummy_csv, "mypipe", "2.0", cfg, tmp_path)
        meta = json.loads(json_p.read_text())

        assert meta["pipeline_name"]    == "mypipe"
        assert meta["pipeline_version"] == "2.0"
        assert meta["config"]           == cfg
        assert meta["config_hash"]      == pa.config_hash(cfg)
        assert meta["output_shape"]     == list(phi.shape)
        assert meta["source_checksum"] is not None    # csv exists on disk
        assert "created_at"   in meta
        assert "numpy_version" in meta

    def test_overwrite_protection_raises(self, tmp_path, dummy_csv, phi):
        kw = dict(csv_path=dummy_csv, pipeline_name="p",
                  pipeline_version="1", config={}, outputs_root=tmp_path)
        pa.save_artifact(phi, **kw)
        with pytest.raises(FileExistsError, match="already exists"):
            pa.save_artifact(phi, **kw)

    def test_allow_overwrite(self, tmp_path, dummy_csv):
        phi1 = np.ones(10)
        phi2 = np.zeros(10)
        kw   = dict(csv_path=dummy_csv, pipeline_name="p",
                    pipeline_version="1", config={}, outputs_root=tmp_path)
        pa.save_artifact(phi1, **kw)
        pa.save_artifact(phi2, **kw, allow_overwrite=True)
        npy_p, _ = pa.artifact_path(dummy_csv, "p", "1", {}, tmp_path)
        assert np.allclose(np.load(npy_p), phi2)

    def test_parent_dir_created_automatically(self, tmp_path, phi):
        csv = tmp_path / "deep" / "nested" / "run.csv"
        csv.parent.mkdir(parents=True)
        csv.write_text("time\n0\n")
        out_root = tmp_path / "outputs"
        npy_p, _ = pa.save_artifact(phi, csv, "p", "1", {}, out_root)
        assert npy_p.exists()


class TestLoadArtifact:

    def test_returns_none_if_missing(self, tmp_path):
        result = pa.load_artifact(
            tmp_path / "nofile.csv", "p", "1", {}, tmp_path
        )
        assert result is None

    def test_returns_array_if_present(self, tmp_path, dummy_csv, phi):
        cfg = {"fs": 50.0}
        pa.save_artifact(phi, dummy_csv, "p", "1", cfg, tmp_path)
        loaded = pa.load_artifact(dummy_csv, "p", "1", cfg, tmp_path)
        assert loaded is not None
        assert np.allclose(phi, loaded)

    def test_wrong_config_returns_none(self, tmp_path, dummy_csv, phi):
        pa.save_artifact(phi, dummy_csv, "p", "1", {"fs": 50.0}, tmp_path)
        result = pa.load_artifact(dummy_csv, "p", "1", {"fs": 100.0}, tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# list_artifacts
# ---------------------------------------------------------------------------

class TestListArtifacts:

    def test_empty_when_no_outputs(self, tmp_path):
        assert pa.list_artifacts(outputs_root=tmp_path) == []

    def test_returns_all_artifacts(self, tmp_path, dummy_csv, phi):
        pa.save_artifact(phi, dummy_csv, "p", "1.0", {"fs": 50.0}, tmp_path)
        pa.save_artifact(phi, dummy_csv, "p", "1.0", {"fs": 100.0}, tmp_path)
        results = pa.list_artifacts(outputs_root=tmp_path)
        assert len(results) == 2

    def test_filtered_by_csv(self, tmp_path, phi):
        csv_a = tmp_path / "runA.csv"; csv_a.write_text("t\n0\n")
        csv_b = tmp_path / "runB.csv"; csv_b.write_text("t\n0\n")
        out   = tmp_path / "out"
        pa.save_artifact(phi, csv_a, "p", "1", {}, out)
        pa.save_artifact(phi, csv_b, "p", "1", {}, out)
        results = pa.list_artifacts(csv_path=csv_a, outputs_root=out)
        assert len(results) == 1
        assert "runA" in results[0]["input_file"]


# ---------------------------------------------------------------------------
# migrate_legacy
# ---------------------------------------------------------------------------

class TestMigrateLegacy:

    def _make_legacy(self, base: Path) -> Path:
        d = base / "npy"; d.mkdir()
        np.save(d / "phi_fast.npy", np.ones(10))
        np.save(d / "phi_subsample.npy", np.arange(5, dtype=float))
        return d

    def test_copies_files(self, tmp_path):
        legacy = self._make_legacy(tmp_path)
        out    = tmp_path / "outputs"
        count  = pa.migrate_legacy(legacy, out)
        assert count == 2
        assert (out / "legacy" / "phi_fast.npy").exists()
        assert (out / "legacy" / "phi_subsample.npy").exists()

    def test_creates_json_stubs(self, tmp_path):
        legacy = self._make_legacy(tmp_path)
        out    = tmp_path / "outputs"
        pa.migrate_legacy(legacy, out)
        meta = json.loads((out / "legacy" / "phi_fast.json").read_text())
        assert meta["pipeline_version"] == "legacy"
        assert "migrated_from" in meta

    def test_no_double_migration(self, tmp_path):
        legacy = self._make_legacy(tmp_path)
        out    = tmp_path / "outputs"
        pa.migrate_legacy(legacy, out)
        count2 = pa.migrate_legacy(legacy, out)
        assert count2 == 0

    def test_originals_preserved(self, tmp_path):
        legacy = self._make_legacy(tmp_path)
        out    = tmp_path / "outputs"
        pa.migrate_legacy(legacy, out)
        assert (legacy / "phi_fast.npy").exists()    # originals untouched

    def test_missing_legacy_dir_returns_zero(self, tmp_path):
        count = pa.migrate_legacy(tmp_path / "nonexistent", tmp_path / "out")
        assert count == 0


# ---------------------------------------------------------------------------
# Text artifacts (URDF / save_text_artifact / load_text_artifact)
# ---------------------------------------------------------------------------

SAMPLE_URDF = '<?xml version="1.0" ?>\n<robot name="test"></robot>\n'


class TestTextArtifacts:

    def test_save_creates_file_and_sidecar(self, tmp_path, dummy_csv):
        art, js = pa.save_text_artifact(
            SAMPLE_URDF, dummy_csv, "phi_to_urdf", "1.0", {"mode": "standalone"},
            suffix=".urdf", outputs_root=tmp_path,
        )
        assert art.exists() and art.suffix == ".urdf"
        assert js.exists()
        assert art.read_text() == SAMPLE_URDF

    def test_roundtrip(self, tmp_path, dummy_csv):
        pa.save_text_artifact(
            SAMPLE_URDF, dummy_csv, "phi_to_urdf", "1.0", {},
            suffix=".urdf", outputs_root=tmp_path,
        )
        loaded = pa.load_text_artifact(
            dummy_csv, "phi_to_urdf", "1.0", {},
            suffix=".urdf", outputs_root=tmp_path,
        )
        assert loaded == SAMPLE_URDF

    def test_load_returns_none_if_missing(self, tmp_path):
        result = pa.load_text_artifact(
            tmp_path / "missing.npy", "p", "1", {}, ".urdf", tmp_path
        )
        assert result is None

    def test_overwrite_protection(self, tmp_path, dummy_csv):
        kw = dict(input_path=dummy_csv, pipeline_name="p", pipeline_version="1",
                  config={}, suffix=".urdf", outputs_root=tmp_path)
        pa.save_text_artifact(SAMPLE_URDF, **kw)
        with pytest.raises(FileExistsError):
            pa.save_text_artifact(SAMPLE_URDF, **kw)

    def test_allow_overwrite(self, tmp_path, dummy_csv):
        kw = dict(input_path=dummy_csv, pipeline_name="p", pipeline_version="1",
                  config={}, suffix=".urdf", outputs_root=tmp_path)
        pa.save_text_artifact("v1", **kw)
        pa.save_text_artifact("v2", **kw, allow_overwrite=True)
        loaded = pa.load_text_artifact(dummy_csv, "p", "1", {}, ".urdf", tmp_path)
        assert loaded == "v2"

    def test_npy_and_urdf_paths_do_not_collide(self, tmp_path, dummy_csv, phi):
        cfg = {"mode": "standalone"}
        npy_p, _  = pa.artifact_path(dummy_csv, "p", "1", cfg, tmp_path, suffix=".npy")
        urdf_p, _ = pa.artifact_path(dummy_csv, "p", "1", cfg, tmp_path, suffix=".urdf")
        assert npy_p != urdf_p
        assert npy_p.stem == urdf_p.stem   # same base name, different suffix

    def test_sidecar_has_content_length(self, tmp_path, dummy_csv):
        _, js = pa.save_text_artifact(
            SAMPLE_URDF, dummy_csv, "p", "1", {}, ".urdf", tmp_path,
        )
        meta = json.loads(js.read_text())
        assert meta["content_length"] == len(SAMPLE_URDF)
        assert "source_checksum" in meta

    def test_different_modes_produce_different_paths(self, tmp_path, dummy_csv):
        p_sa, _ = pa.artifact_path(
            dummy_csv, "p", "1", {"mode": "standalone"}, tmp_path, ".urdf"
        )
        p_tp, _ = pa.artifact_path(
            dummy_csv, "p", "1", {"mode": "template"}, tmp_path, ".urdf"
        )
        assert p_sa != p_tp


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------

class TestAtomicWrite:

    def test_no_partial_file_on_oserror(self, tmp_path, dummy_csv, phi, monkeypatch):
        """If the rename fails the temp file should be cleaned up."""
        import pipeline_artifacts as _pa

        original = _pa.Path.replace

        call_count = {"n": 0}

        def failing_replace(self, target):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated disk full")
            return original(self, target)

        monkeypatch.setattr(_pa.Path, "replace", failing_replace)

        with pytest.raises(OSError):
            pa.save_artifact(phi, dummy_csv, "p", "1", {}, tmp_path)

        # No stray temp files should remain.
        leftover = list((tmp_path / dummy_csv.stem).glob("*.tmp"))
        assert leftover == []
