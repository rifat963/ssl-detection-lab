from pathlib import Path

import pytest

import ssldet.video as video
from ssldet.catalog import TRACKERS, available_tracker_names, capabilities, resolve_tracker
from ssldet.video import VideoAnalysisConfig


def _config(**overrides):
    values = {
        "video_source": "match.mp4",
        "model_name": "yolo26n",
        "weights_file": "best.pt",
    }
    values.update(overrides)
    return VideoAnalysisConfig(**values)


def test_catalog_lists_every_ultralytics_tracker():
    assert set(available_tracker_names()) == {
        "botsort",
        "bytetrack",
        "ocsort",
        "deepocsort",
        "fasttrack",
        "tracktrack",
    }
    assert len(capabilities()["trackers"]) == len(TRACKERS)
    assert all(item.config_file == f"{item.name}.yaml" for item in TRACKERS)


def test_tracker_resolution_accepts_names_and_config_filenames():
    assert resolve_tracker("bytetrack").name == "bytetrack"
    assert resolve_tracker("bytetrack.yaml").name == "bytetrack"
    assert resolve_tracker("ByteTrack.YAML").name == "bytetrack"


def test_unknown_tracker_is_rejected_during_configuration():
    with pytest.raises(ValueError, match="Unknown tracker"):
        resolve_tracker("bytrack.yaml")

    # The failure must surface from validate(), before any weights or video are touched.
    with pytest.raises(ValueError, match="Unknown tracker"):
        _config(tracker="bytrack.yaml").validate()


def test_builtin_and_custom_trackers_both_validate(tmp_path):
    assert _config(tracker="deepocsort.yaml").validate().tracker == "deepocsort.yaml"
    assert _config(tracker=None).validate().tracker is None

    custom = tmp_path / "my_botsort.yaml"
    custom.write_text("tracker_type: botsort\ntrack_buffer: 90\n", encoding="utf-8")
    assert _config(tracker=str(custom)).validate().tracker == str(custom)


def test_tracker_settings_capture_a_custom_config_for_reproducibility(tmp_path, monkeypatch):
    custom = tmp_path / "my_botsort.yaml"
    custom.write_text("tracker_type: botsort\ntrack_buffer: 90\n", encoding="utf-8")
    monkeypatch.setattr(video, "_tracker_config_path", Path)

    settings = video._tracker_settings(str(custom))

    assert settings["enabled"] is True
    assert settings["resolved"] is True
    assert settings["tracker_type"] == "botsort"
    assert settings["is_builtin"] is False
    assert settings["settings"]["track_buffer"] == 90


def test_tracker_settings_report_disabled_tracking():
    settings = video._tracker_settings(None)

    assert settings == {"enabled": False, "tracker": None, "resolved": False, "settings": {}}


def test_catalog_matches_the_installed_ultralytics_tracker_set():
    """Fail loudly when an Ultralytics upgrade adds or renames a tracker.

    The catalog is static data so it can render before Ultralytics loads, which means it
    can drift. This is the guard that turns that drift into a test failure instead of a
    tracker students never discover.
    """

    cfg = pytest.importorskip("ultralytics.cfg", reason="Ultralytics not installed")
    shipped = {path.stem for path in (Path(cfg.__file__).parent / "trackers").glob("*.yaml")}

    assert shipped == set(available_tracker_names()), (
        "ssldet.catalog.TRACKERS is out of sync with the installed Ultralytics. "
        f"Shipped: {sorted(shipped)}; catalogued: {sorted(available_tracker_names())}."
    )


def test_tracker_settings_survive_an_unresolvable_config(monkeypatch):
    def explode(value):
        raise OSError("ultralytics could not locate the tracker file")

    monkeypatch.setattr(video, "_tracker_config_path", explode)

    settings = video._tracker_settings("botsort.yaml")

    # A reporting failure must not discard an analysis that already succeeded.
    assert settings["enabled"] is True
    assert settings["resolved"] is False
    assert settings["catalog_name"] == "botsort"
    assert settings["tracker_type"] == "botsort"
