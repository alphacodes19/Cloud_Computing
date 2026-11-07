# CS3081 Mini Project — Command-Parser Knowledge Distillation

<ROLL_NUMBER> — <YOUR NAME>

## Repository layout
```
cluster/        k3d cluster config (Task 1)
prompts/        teacher prompt (Task 1)
worker/         sharded teacher-labelling worker + clean.py (Task 1)
training/       train.py, predict.py, config.yaml (Task 2)
service/        FastAPI inference service + Dockerfile (Task 3)
k8s/            Deployment/Service/Ingress/HPA/Secret manifests (Task 4)
network/        subnets.md, nacl.json (Task 5)
.github/workflows/ci.yml   CI pipeline (Task 6)
results/        score.py / Locust / kubectl output committed as evidence
report/         report.pdf, this project's write-up (Task 1-7)
tests/          unit tests
```

## Build, deploy, test, teardown

```bash
# 1. stand up the cluster, SeaweedFS, and the service
make up

# 2. train the student model (writes models/v1, training/loss.csv)
make train                 # full run
make train SUBSET=50       # quick smoke test only

# 3. score a model's predictions against validation.jsonl
python training/predict.py --model-dir models/v1 --out results/student_predictions.jsonl
python tools/score.py results/student_predictions.jsonl data/validation.jsonl --out results/student_report.json

# 4. load test (through the Ingress, once the Deployment is Ready)
pip install locust
locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m \
    --host http://localhost:8080 --csv results/loadtest

# 5. verify the NACL
python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check

# 6. tear down
make down
```

## GitHub links
- Repository: <PASTE LINK>
- Model release (v1): <PASTE LINK>
- Model release (v2): <PASTE LINK>
- CI run that fails the quality gate: <PASTE LINK>
