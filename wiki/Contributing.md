# Contributing

## Development setup

```bash
git clone https://github.com/rifat963/ssl-detection-lab.git
cd ssl-detection-lab
pip install -e ".[dev]"
```

## Checks before every PR

```bash
pytest -q
ruff check src tests examples
python -m compileall -q src tests
```

All three must pass. Ruff is configured in `pyproject.toml` (line length 100, target py310,
rules `E4 E7 E9 F B I`).

> The `output/` notebooks are intentionally **not** ruff-clean — notebooks legitimately import
> after an install cell (`E402`) and re-import for cell independence. Scope lint runs to
> `src tests examples`.

## Adding an SSL objective

1. **Subclass `SSLMethod`** in `src/ssldet/methods/your_method.py`:

```python
from .common import SSLMethod, ema_update, frozen_copy

class YourMethod(SSLMethod):
    requires_two_views = True          # omit or False for single-view objectives

    def __init__(self, encoder, feature_dim, **kwargs):
        super().__init__()
        self.online_encoder = encoder  # name it online_encoder or student_encoder
        ...

    def forward(self, batch):
        return loss                    # a scalar tensor

    @torch.no_grad()
    def after_optimizer_step(self):    # optional: EMA updates go here
        ema_update(self.online_encoder, self.target_encoder, self.current_momentum)
```

**Naming matters.** The encoder attribute must be `online_encoder` or `student_encoder` — that
is how [Downstream Transfer](Downstream-Transfer.md) finds the weights to move into a detector.

**`requires_two_views`** is read by the trainer via `ssl_module_requires_two_views()` to decide
whether the dataset is large enough. Set it rather than hard-coding a check.

2. **Export it** from `methods/__init__.py` and register it in `ssl/factory.py`.

3. **Add the method name** to the `supported` set in `PretrainConfig.validate()`, along with any
   parameter constraints, and to `SSL_ARCHITECTURES` in `catalog.py`.

   These three must stay in sync — `tests/test_ssl_factory.py` asserts that the registry, the
   catalog, and the config's supported set are identical.

4. **Wire the constructor arguments** in `registry.py::build_method`.

5. **Add tests.** At minimum: the loss is finite and backpropagates, and any EMA target receives
   no gradients.

Third-party objectives can register at runtime without touching the package:

```python
from ssldet.ssl import register_ssl_module
register_ssl_module("mymethod", MyMethod)
```

## Adding a detection backend

Implement the `ObjectDetector` protocol (`predict`, `track`, `validate`) and register it:

```python
from ssldet.detection import register_detection_backend
register_detection_backend("mybackend", MyDetector.load)
```

## Test conventions

Tests are pure-Python and must not download weights or require a GPU. Use fakes for Ultralytics
metrics objects — see `tests/test_reporting.py` for the established pattern.

A regression test should **fail against the unfixed code**. Verify that by reverting the fix
locally before you commit; a test that passes either way is not protecting anything.

## Notebook conventions

- Live in `output/yolo26-notebooks/` or `output/yolo12-notebooks/`
- Assert the package version floor in the setup cell, and keep that floor current
- Use official model spellings: `yolo26n.yaml`, `yolo12n.yaml` — never `yolov26`/`yolov12`
- Edit the raw JSON surgically. A `json.load`/`json.dump` round-trip can reformat the whole file
  and bury a one-line change in a thousand-line diff
- State clearly that SSL loss is not a detection metric

## Documentation

Wiki pages follow [Wiki Guidelines](Wiki-Guidelines.md). Update the wiki in the same PR as the
behaviour change — documentation that lags the code is worse than none.

## Honesty requirements

This is teaching material. Implementations that adapt a published method must say so:

- Keep the **adaptation notices** on MAE, DINOv2, and I-JEPA
- Never describe a compute-scaled adaptation as a reproduction
- Never present SSL loss as evidence of detection quality
- Preserve the labelled/unlabelled metric boundary in evaluation and video reports

## Licensing

Contributions are MIT. Do not vendor AGPL, DINOv3-licensed, or other restrictively licensed code
or weights. New dependencies must be recorded in `THIRD_PARTY_NOTICES.md`.
