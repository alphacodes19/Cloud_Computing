# Task 6 — Reproducibility, testing, and CI/CD

This document records reproducible commands and evidence for Task 6. Results below distinguish existing GitHub evidence from steps that still need to be completed for the final documentation commit.

## Reproduction environment and artifacts

- Python: 3.11 (`.python-version`; CI uses 3.11).
- Training/inference model stack: PyTorch 2.3.1 CPU and Transformers 4.42.3. The training dependency pins are in `training/requirements-training.txt`; the inference service pins are in `service/requirements.txt`.
- Training parameters and seed: `training/config.yaml`. The recorded v1 and v2 runs used seed 42, 1,500 training records, and 3 and 4 epochs respectively; see `results/training-v1-metadata.json` and `results/training-v2-metadata.json`.
- The model archives are published in the [`model-v1` GitHub release](https://github.com/alphacodes19/Cloud_Computing/releases/tag/model-v1). `results/model-artifact-checksums.txt` records the separately checked SeaweedFS model-file digests; the release contains `SHA256SUMS` for its archives.
- Training data under `datasets/` is ignored by Git and is not included in the repository. Reproducing model training requires regenerating the cleaned teacher-labelled training data through the Task 1 storage/label workflow before running `make train`. Do not interpret a fresh clone alone as containing the training dataset.

### Local setup and verification

From a fresh clone, install the pinned training dependencies if training is required, restore/regenerate the dataset, then run:

```bash
python -m pip install -r training/requirements-training.txt
make train SUBSET=50                  # quick smoke test; requires datasets/v1/train.jsonl
make train                            # full configured run
python training/predict.py --model-dir models/v1 --validation data/validation.jsonl --out results/student_predictions.jsonl
python tools/score.py results/student_predictions.jsonl data/validation.jsonl --out results/student_report.json
python -m pip install pytest ruff
ruff check --select E4,E7,E9,F worker training service tests
pytest tests/ -v
```

A full `make up`/`make down` cluster lifecycle is documented in the README and requires local k3d, kubectl, Docker, and configured model-storage credentials. Kubernetes credentials belong in the ignored local `k8s/secret.yaml`, created from `k8s/secret.example.yaml`; never commit real credentials.

## CI stages

The workflow is [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

1. **lint** — Ruff checks syntax-related and common-error rules in worker, training, service, and tests.
2. **unit-tests** — installs CPU PyTorch and service dependencies, then runs `pytest tests/ -v`.
3. **evaluate-model** — downloads the `model-v1` release, verifies its archive checksums, extracts the v1 model, runs validation predictions (or uses the checked-in untrained fixture in the explicit demonstration mode), scores them, and uploads predictions/report as an artifact.
4. **quality-gate** — fails if prediction JSON validity on `data/validation.jsonl` is below 95%.
5. **secret-scan** — runs Gitleaks over the checked-out Git history.

The manual workflow-dispatch input `test_quality_gate_failure=true` intentionally selects `results/evaluation/pythia_untrained_predictions.jsonl`. It should make only the **quality-gate job** fail; that red result is the expected demonstration, not a passing CI run.

## CI and security evidence

- **Successful baseline CI:** [run 42526703693](https://github.com/alphacodes19/Cloud_Computing/actions/runs/42526703693) completed successfully on the earlier baseline commit. A successful workflow run means its jobs, including the Gitleaks `secret-scan`, completed successfully for that commit. This is historical evidence, not a result for the documentation changes in this branch.
- **Deliberate gate failure:** [run 42526718020](https://github.com/alphacodes19/Cloud_Computing/actions/runs/42526718020) demonstrates the intended failure. The quality-gate log reported `json_validity 0.0 < 95.0`, `examples: 100`, and exited with code 1. Do not cite this run as a CI success.
- **Documentation branch:** the CI run triggered by the Task 6 documentation commit must finish before this work is treated as verified. Link that run here after it completes. Confirm the `secret-scan` job result in the run summary before describing the security scan as passing for the final commit.
- Gitleaks is automated evidence of secret scanning, not proof of overall security. Keep real cloud credentials out of tracked files and rotate any credential ever exposed in Git history.

## Final reproducible Git tag

Create an annotated tag only after the Task 6 changes have been merged to the intended final branch and that exact commit has a successful CI run. First check whether the proposed tag already exists; choose a different unique name if it does.

```bash
git checkout main
git pull --ff-only origin main
git status --short                    # should be clean
git log -1 --oneline
git ls-remote --tags origin refs/tags/task6-final
git tag -a task6-final -m "Task 6 final: reproducibility and CI evidence"
git push origin task6-final
git show --no-patch --decorate task6-final
```

If `git ls-remote` shows that `task6-final` already exists, do not overwrite or force-update it. Select a new unique final tag name, check it, then create and push that tag. Record the resulting tag URL and commit SHA here and in the README. A tag has **not** been created as part of this documentation change; it must point to the verified final commit, not a speculative or pre-merge branch commit.
