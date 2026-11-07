.PHONY: up down train

# Builds/deploys the whole system from a fresh clone, in <=3 commands.
up:
	k3d cluster create --config cluster/k3d.yaml
	kubectl apply -f k8s/00-namespaces.yaml
	bash scripts/deploy_all.sh   # SeaweedFS, image build+import, secret, deployment/service/ingress/hpa

down:
	k3d cluster delete cs3081-cluster

# make train            -> full run per training/config.yaml
# make train SUBSET=50  -> quick smoke test on the first 50 records
train:
	python training/train.py --config training/config.yaml --out models/v1 $(if $(SUBSET),--subset $(SUBSET),)
