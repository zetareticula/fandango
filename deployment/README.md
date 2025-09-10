# Fandango Deployment Orchestration

This directory contains an OCaml-based deployment orchestration system for Fandango, providing composable and factorized deployment strategies across different environments.

## Features

- **Environment-aware deployments**: Dev, Staging, Production configurations
- **Multiple deployment strategies**: Kubernetes, Docker, Local
- **Pipeline composition**: Chain multiple deployment steps
- **Rollback capabilities**: Safe deployment rollbacks
- **Health monitoring**: Deployment status checking

## Architecture

The deployment system is built using OCaml and Dune, providing:

1. **Modular Strategies**: Each deployment target (K8s, Docker, Local) is implemented as a separate strategy
2. **Configuration Factorization**: Environment-specific configurations are automatically generated
3. **Pipeline Composition**: Multiple strategies can be chained together
4. **Type Safety**: OCaml's type system ensures deployment configuration correctness

## Installation

```bash
# Install OCaml and Dune
opam install dune lwt lwt_ppx yojson cmdliner logs fmt

# Build the deployment tool
cd deployment
dune build

# Install the binary
dune install
```

## Usage

### Deploy to a single environment

```bash
# Deploy to local development
fandango-deploy deploy --env dev --strategy local --target localhost

# Deploy to Kubernetes staging
fandango-deploy deploy --env staging --strategy kubernetes --target staging-cluster

# Deploy to production
fandango-deploy deploy --env production --strategy kubernetes --target prod-cluster
```

### Run deployment pipeline

```bash
# Run multi-stage pipeline
fandango-deploy pipeline --config pipeline.json
```

### Check deployment status

```bash
fandango-deploy status
```

## Configuration

### Environment Configurations

- **Dev**: 1 replica, 250m CPU, 256Mi memory
- **Staging**: 2 replicas, 500m CPU, 512Mi memory  
- **Production**: 3 replicas, 1000m CPU, 1Gi memory

### Deployment Strategies

1. **Kubernetes Strategy**
   - Generates K8s manifests
   - Applies deployments via kubectl
   - Supports rollbacks

2. **Docker Strategy**
   - Builds Docker images
   - Pushes to registry
   - Tags by environment

3. **Local Strategy**
   - Runs Fandango locally
   - Development testing
   - Port configuration

## Pipeline Example

```ocaml
let production_pipeline = create_pipeline [
  docker_strategy;      (* Build and push image *)
  kubernetes_strategy;  (* Deploy to K8s *)
]
```

## Integration with Fandango

The deployment system integrates with:

- Fandango quantization server
- Kubernetes manifests in `/k8s`
- Docker configuration
- Health check endpoints

## Monitoring

The system provides:

- Health check integration (`/health` endpoint)
- Deployment status monitoring
- Kubernetes deployment tracking
- Local server status checking
