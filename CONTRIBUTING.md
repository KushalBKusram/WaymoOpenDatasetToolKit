# Contributing

Thanks for looking at the toolkit. This is a small research codebase; keep changes focused and
reproducible.

## Setup and tests

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or docs/requirements-tested.txt for the audited versions
pip install pytest ruff
python -m pytest -q                    # offline, no GPU, no Waymo data
```

The tests stub torch, the detector registry and the GCS reader (see `tests/conftest.py`), so they run
without credentials. CI (`.github/workflows/ci.yml`) installs only numpy, scipy, pandas, motmetrics,
pyyaml, opencv, pytest and ruff, runs a syntax-level `ruff` check and then the tests. Anything that needs
the dataset, a GPU or a trained model is not covered by CI and should be smoke-tested by hand (see the
README's smoke-test commands).

## Adding a detector

Follow the "Extending the detector registry" section of the README: subclass `BaseDetector` in
`models/<name>_detector.py`, decorate it with `@register_detector('Name')`, import it from
`models/__init__.py`, add `configs/<name>.yaml`, and run a one-frame smoke test with a fresh `--drive-dir`
before scaling up.

## Reporting results

Numbers that go into the README or `docs/` must follow
[docs/benchmark_protocol.md](docs/benchmark_protocol.md): the frozen segment manifests, the stated setting
(S or L), the metric definitions, and more than one seed before a difference is claimed (the protocol records
the measured seed spread). State what was not run. Do not report runs made with `--max-frames` or from older
protocol versions.

## What not to commit

- Waymo data, the `.waymo_cache/` directory, or anything under `runs/`.
- Model checkpoints (`*.pt`), detection dumps (`*.npz`) or credentials and access tokens.
- Large images. Any frame taken from the Waymo Open Dataset must carry an attribution to the dataset and
  respect the [Waymo Open Dataset license terms](https://waymo.com/open/terms).

## License

The code is licensed under the GNU AGPL v3 (see `LICENSE`); contributions are accepted under the same terms.
