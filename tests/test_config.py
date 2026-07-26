from ssldet.config import PretrainConfig


def test_all_supported_methods_validate():
    for method in ["simclr", "byol", "moco", "dinov2", "dinov3", "mae", "ijepa"]:
        values = {"dinov3_weights": "teacher.pth"} if method == "dinov3" else {}
        config = PretrainConfig(method=method, image_roots=["unused"], **values)
        assert config.validate().method == method


def test_ijepa_projection_dimension_constraint():
    config = PretrainConfig(method="ijepa", image_roots=["unused"], projection_dim=258)
    try:
        config.validate()
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Invalid I-JEPA projection dimension was accepted")


def test_unrelated_ijepa_settings_do_not_reject_other_methods():
    config = PretrainConfig(
        method="simclr",
        image_roots=["unused"],
        projection_dim=258,
        predictor_heads=4,
    )

    assert config.validate().projection_dim == 258


def test_student_input_errors_are_rejected_before_training():
    invalid_values = (
        {"image_roots": "images"},
        {"max_images": 0},
        {"workers": -1},
        {"learning_rate": 0.0},
        {"warmup_epochs": 26},
    )
    for values in invalid_values:
        try:
            arguments = {"method": "simclr", "image_roots": ["unused"], **values}
            config = PretrainConfig(**arguments)
            config.validate()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Invalid configuration was accepted: {values}")
