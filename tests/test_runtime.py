from ssldet.runtime import assert_supported_runtime, runtime_report


def test_runtime_report_has_version_and_accelerator_sections():
    report = runtime_report()

    assert set(report["packages"]) == {"torch", "torchvision", "ultralytics"}
    assert {"available", "runtime", "cudnn", "device_count", "devices"} <= set(
        report["cuda"]
    )
    assert report["features"]["dinov3"]["minimum_torch"] == "2.7.1"


def test_runtime_assertion_reports_missing_gpu(monkeypatch):
    report = runtime_report()
    report["supported"] = True
    for values in report["packages"].values():
        values["supported"] = True
    report["cuda"]["available"] = False
    report["cuda"]["device_count"] = 0
    monkeypatch.setattr("ssldet.runtime.runtime_report", lambda: report)

    try:
        assert_supported_runtime(require_cuda=True, minimum_gpus=2)
    except RuntimeError as error:
        assert "CUDA is unavailable" in str(error)
        assert "requires at least 2" in str(error)
    else:
        raise AssertionError("Missing CUDA requirements were accepted")
