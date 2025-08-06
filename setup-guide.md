# limps-suite GitLab CI/CD Setup Guide

This guide will walk you through deploying the complete CI/CD infrastructure to your GitLab repositories.

## Prerequisites

1. **All repositories cloned locally** in a common directory (e.g., `~/code/`)
2. **GitLab access** with permissions to create groups and projects
3. **GitLab Container Registry** enabled
4. **GitLab runners** registered (Docker + optional GPU)

## Step 1: Prepare Your Repository Structure

Ensure you have all repositories cloned:

```bash
~/code/
├── 9xdSq-LIMPS-FemTO-R1C/
├── symbolic-polynomial-server/
├── entropy-engine/
├── LiMp/
├── motif-detection/
├── poly-optimizer-client/
├── Choppy-Backend/
├── Choppy-Frontend/
└── limps-infra/
```

## Step 2: Run the Deployment Script

```bash
# Make the script executable
chmod +x deploy-to-repos.sh

# Run the deployment (specify your repo directory)
./deploy-to-repos.sh ~/code
```

The script will:
- ✅ Check all repositories exist
- 📦 Copy CI/CD files to each repository
- 🔄 Create git branches (`add-gitlab-ci` or `add-orchestrator`)
- 💾 Commit changes with descriptive messages

## Step 3: Push All Branches

```bash
# Push all branches to GitLab
for repo in ~/code/*; do
    echo "Pushing $repo..."
    cd "$repo"
    if git branch | grep -q "add-gitlab-ci"; then
        git push origin add-gitlab-ci
    elif git branch | grep -q "add-orchestrator"; then
        git push origin add-orchestrator
    fi
done
```

## Step 4: Create GitLab Groups and Projects

### Create Group Structure
```
limps-suite/
├── core/
├── services/
├── apps/
└── infra/
```

### Import Repositories
Import your GitHub repositories to these exact GitLab paths:

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

## Step 5: Configure GitLab Variables

In the `limps-suite` group → Settings → CI/CD → Variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `PYPI_TOKEN` | PyPI API token for publishing | `pypi-xxxxxxxxxxxxxxxx` |
| `STAGING_URL` | Staging environment URL | `https://staging.example.com` |

*Note: GitLab automatically provides `CI_REGISTRY_USER` and `CI_REGISTRY_PASSWORD`*

## Step 6: Create Merge Requests

Create merge requests **in dependency order**:

### 1. Core Services (First)
- `9xdSq-LIMPS-FemTO-R1C` → `main`
- `symbolic-polynomial-server` → `main`
- `entropy-engine` → `main`

### 2. Service Layer
- `LiMp` → `main`
- `motif-detection` → `main`
- `poly-optimizer-client` → `main`

### 3. Applications
- `Choppy-Backend` → `main`
- `Choppy-Frontend` → `main`

### 4. Infrastructure (Last)
- `limps-infra` → `main`

## Step 7: Merge and Deploy

1. **Merge core services first** - these have no dependencies
2. **Merge service layer** - depends on core services
3. **Merge applications** - depend on services
4. **Merge infrastructure last** - triggers orchestrator pipeline

## Step 8: Verify Deployment

After merging `limps-infra`:

1. **Check orchestrator pipeline** in `limps-suite/infra`
2. **Monitor build order** - should follow dependency graph
3. **Verify Docker images** are pushed to registry
4. **Check staging deployment** via `deploy.sh`

## Architecture Overview

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

## Troubleshooting

### Build Failures
- Check runner configuration
- Verify Docker registry access
- Ensure all dependencies are available

### Dependency Errors
- Verify merge order (core → services → apps → infra)
- Check orchestrator pipeline configuration
- Ensure all required images are built

### Deployment Issues
- Verify `STAGING_URL` is set correctly
- Check Docker Compose configuration
- Ensure all services are healthy

## Next Steps

Once deployed:
1. **Monitor pipelines** for any issues
2. **Set up monitoring** for production deployment
3. **Configure alerts** for build failures
4. **Scale runners** as needed for your workload

## Support

If you encounter issues:
1. Check GitLab CI/CD logs
2. Verify all variables are set correctly
3. Ensure runners have sufficient resources
4. Review dependency graph for build order issues