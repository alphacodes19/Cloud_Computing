#!/usr/bin/env bash
# Second of the <=3 `make up` commands: everything after the cluster + namespaces exist.
# Fill in the SeaweedFS install + bucket creation for your chosen approach (Helm chart or
# a plain Deployment+Service manifest) before relying on this in submission — the exact
# commands depend on which SeaweedFS install method you pick.
set -euo pipefail

echo "1) Deploying SeaweedFS into namespace 'data' ..."
# kubectl apply -n data -f k8s/seaweedfs.yaml   # <- add this manifest yourself
# then create the two buckets (datasets, models) via the SeaweedFS S3 API or 'weed shell'

echo "2) Building and importing the service image ..."
docker build -t parser-service:latest -f service/Dockerfile .
k3d image import parser-service:latest -c cs3081-cluster

echo "3) Applying the Secret, Deployment, Service, Ingress, HPA ..."
kubectl apply -f k8s/secret.yaml   # copy from k8s/secret.example.yaml and fill in real values first
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

echo "Waiting for rollout..."
kubectl -n app rollout status deployment/parser-service
