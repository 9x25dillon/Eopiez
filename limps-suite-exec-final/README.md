# limps-suite CI/CD Infrastructure Setup

This package contains **ready-to-run GitLab CI/CD** configs, Dockerfiles, and deployment infrastructure for the limps-suite microservices platform.

## Quick Start Guide

### 1. Create GitLab Group Structure

Create the following group hierarchy in GitLab:
```
limps-suite/
├── core/
├── services/
├── apps/
└── infra/
```

### 2. Import GitHub Repositories

Import your existing GitHub repositories to these exact GitLab paths:

| GitHub Repository | GitLab Project Path |
|-------------------|---------------------|
| `9xdSq-LIMPS-FemTO-R1C` | `limps-suite/core/limps-matrix-optimizer` |
| `symbolic-polynomial-server` | `limps-suite/core/symbolic-polynomial-svc` |
| `entropy-engine` | `limps-suite/core/entropy-engine` |
| `LiMp` | `limps-suite/services/al-uls-orchestrator` |
| `motif-detection` | `limps-suite/services/motif-detection` |
| `poly-optimizer-client` | `limps-suite/services/poly-optimizer-client` |
| `Choppy-Backend` | `limps-suite/apps/choppy-backend` |
| `Choppy-Frontend` | `limps-suite/apps/choppy-frontend` |
| `limps-infra` | `limps-suite/infra` |

### 3. Copy CI/CD Files

For each project, copy the corresponding files from this package:

- **Individual projects**: Copy `.gitlab-ci.yml` and `Dockerfile` from the matching folder
- **Infra project**: Copy the entire `infra/` folder contents

### 4. Configure GitLab Variables

In the `limps-suite` group → Settings → CI/CD → Variables, set:

- `PYPI_TOKEN` - For publishing Python packages to PyPI
- `STAGING_URL` - Your staging environment URL (e.g., https://staging.example.com)

*Note: GitLab automatically provides `CI_REGISTRY_USER` and `CI_REGISTRY_PASSWORD` when Container Registry is enabled.*

### 5. Set Up Infrastructure

1. **Enable GitLab Container Registry** for the group
2. **Register runners**:
   - Docker runner (required)
   - GPU runner (optional, for Julia/CUDA workloads)

### 6. Deploy

1. **Push to any project** → builds, tests, and pushes Docker image
2. **Run orchestrator pipeline** in `limps-suite/infra` → builds in dependency order → deploys via Docker Compose

## Architecture Overview

### Service Dependencies
```
core:limps-matrix-optimizer
├── core:symbolic-polynomial-svc
├── core:entropy-engine
├── services:motif-detection
└── services:poly-optimizer-client
    └── services:al-uls-orchestrator
        └── apps:choppy-backend
            └── apps:choppy-frontend
```

### Technology Stack
- **Core Services**: Julia (matrix optimization, polynomial processing, entropy calculations)
- **Service Layer**: Python (orchestration, client libraries)
- **Applications**: Python FastAPI (backend), Node.js/React (frontend)
- **Infrastructure**: Docker Compose, GitLab CI/CD

## File Structure

```
limps-suite-exec-final/
├── PROJECT_MAPPING.json          # Repository mappings
├── infra/                        # Infrastructure & deployment
│   ├── ci-templates/
│   │   └── orchestrator.gitlab-ci.yml
│   ├── deploy-manifests/
│   │   ├── docker-compose.yml
│   │   └── scripts/
│   │       └── deploy.sh
│   └── init.sh
├── core/                         # Core algorithmic services
├── services/                     # Business logic services
├── apps/                         # User-facing applications
└── README.md                     # This file
```

## Troubleshooting

- **Build failures**: Check runner configuration and Docker registry access
- **Deployment issues**: Verify `STAGING_URL` and Docker Compose setup
- **Dependency errors**: Ensure orchestrator pipeline runs in correct order