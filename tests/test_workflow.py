from types import SimpleNamespace

from ssldet.workflow import launch_distributed_pretrain, make_dry_run_config


def test_make_dry_run_config_uses_student_safe_defaults(tmp_path):
    config = make_dry_run_config(
        "dinov2",
        tmp_path / "train" / "images",
        tmp_path / "run",
        yolo_model="yolo11n.yaml",
    )

    assert config.method == "dinov2"
    assert config.epochs == 1
    assert config.max_images == 32
    assert config.image_size == 128
    assert config.yolo_model == "yolo11n.yaml"


def test_distributed_launcher_writes_yaml_and_builds_torchrun_command(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "run"
    config = make_dry_run_config("simclr", tmp_path / "images", output_dir)
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        (output_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("ssldet.workflow.subprocess.run", fake_run)
    result = launch_distributed_pretrain(config, num_processes=2)

    assert result.succeeded
    assert result.config_path.exists()
    assert "--nproc_per_node=2" in captured["command"]
    assert captured["check"] is False
