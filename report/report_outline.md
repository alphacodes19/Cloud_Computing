# Report outline (report/report.pdf, max 8 pages excl. references/appendices)

Write this as the real report once you have real numbers — do not invent any
measured value. One section per task; keep "measured" and "assumed" visually
separate (e.g. a labelled "Measured" box vs "Assumption" box, or two columns).

## Task 1 — Object Storage and Parallel Teacher Labelling
- Cluster/SeaweedFS setup, bucket names, sharding approach (your real SHARD_COUNT)
- Counts table (input / bad_json / bad_keys / bad_intent / bad_slot / duplicates / kept)
  straight from `python worker/clean.py ... ` output — paste it, don't retype it
- Sharding explanation (half page)

## Task 2 — Training the Student
- config.yaml values, loss.csv curve (one line/table)
- models/v1 vs models/v2 — what differs
- Side-by-side score.py table: teacher vs untrained pythia-70m vs your student,
  on validation.jsonl (from results/*.json — paste the real numbers)
- Sequence-level vs logit-based distillation explanation (one page)

## Task 3 — Inference Service in a Container
- Endpoint behaviour, FAULT_INJECT test evidence
- Image size (`docker images`), dependency pinning approach
- Peak memory under 1 CPU / 1 GiB for 50 sequential requests — real number + command output

## Task 4 — Kubernetes Deployment, Scaling, Reliability
- Manifests summary; HPA behaviour observed (screenshot/log of 1->2->1 replicas)
- Rolling update + rollback evidence (X-Model-Version before/after)
- Pod-delete recovery evidence
- Three cold-start timings + median, with log evidence of model download
- Load-balancing: % of requests per replica from loadtest_headers.json
- State your single-laptop/single-node limitation explicitly

## Task 5 — Subnetting and NACL
- Table from network/subnets.md
- nacl_sim.py --check output against the hidden 20-packet set (you only have the
  6-packet sample; describe how you tested beyond it)
- Stateless-NACL / rule-order explanation

## Task 6 — Reproducibility, Testing, CI/CD
- `make up` / `make down` command count
- CI stages and what each checks
- Link to the deliberately-failing run (bad model)
- gitleaks result

## Task 7 — Performance, Cost, Review
- p50/p95 latency, throughput, error rate, kubectl top pods, cold-start — from your
  real Locust + kubectl output
- Cost-per-1000-requests worked example using data/rate_card.json and your actual
  replica count/limits and measured requests/hour
- SLO + error budget (target, window, requests)
- Review table: one measured number per row (Data/storage, Model quality, Latency,
  Resource utilisation, Reliability, Security/cost)
