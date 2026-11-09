# AI_USE.md

## Scope and disclosure

For the Task 6 documentation pass, GitHub Copilot was used to inspect the existing repository files and CI workflow, identify unfinished Task 6 documentation, and help draft reproducibility and security-evidence text. The suggestions were checked against the workflow, scoring script, checked-in metadata/checksums, and available GitHub Actions run records before being added.

This file records only the AI assistance that can be verified for this pass. It does not claim that a particular AI tool was or was not used for Tasks 1–5 or 7. The project author should add those details from their own records before submitting; do not copy in tool names or activities that did not actually occur.

## Task 6 assistance and verification

- **Asked for:** help completing the remaining Task 6 work: CI/CD reproducibility instructions, a truthful AI-use disclosure, security evidence, and final-tag instructions.
- **AI-assisted:** repository inspection and drafting/editing of Task 6 documentation. The existing CI workflow and its deliberate quality-gate test were reviewed; this documentation pass does not claim to have authored the historical application, model, or infrastructure code.
- **Checked against:** `.github/workflows/ci.yml`, `tools/score.py`, the checked-in untrained prediction fixture, model artifact checksums, and the linked GitHub Actions runs.
- **Observed results:** the existing normal CI run completed successfully; the deliberate quality-gate run failed at the intended 95% JSON-validity check (0.0%, 100 examples, exit code 1). Links and scope are recorded in `TASK6_REPRODUCIBILITY.md`.
- **Limitations:** no local shell, cluster, or Git tag-push operation was available during this pass. The documentation commit's CI must finish on GitHub, and the final tag must be created after merging the verified commit to the intended release branch.

## Tasks 1–5 and 7

The historical AI tools, prompts, accepted/reworked suggestions, and independent verification for these tasks are not established by this documentation pass. The project author must complete this section accurately from their own work history. The checked-in evaluation and performance numbers should remain sourced from actual result files/logs, not generated or estimated by AI.

## Work not delegated to AI

No measured metric is asserted here without repository evidence. The report's interpretations, reflections, and any claims about work performed personally must be reviewed and written by the project author in their own words.
