from ssldet.config import PretrainConfig


def test_all_supported_methods_validate():
    for method in ["simclr", "byol", "moco", "mae", "ijepa"]:
        config = PretrainConfig(method=method, image_roots=["unused"])
        assert config.validate().method == method


def test_ijepa_projection_dimension_constraint():
    config = PretrainConfig(method="ijepa", image_roots=["unused"], projection_dim=258)
    try:
        config.validate()
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Invalid I-JEPA projection dimension was accepted")

