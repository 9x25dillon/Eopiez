# GitLab Migration Guide for LIMPS Suite

This guide will help you migrate your GitHub repositories to GitLab and set up a comprehensive CI/CD pipeline for your LIMPS suite.

## Overview

The migration will create the following structure in GitLab:

```
limps-suite/
│
├── core/
│   ├── limps-matrix-optimizer
│   ├── symbolic-polynomial-svc
│   └── entropy-engine
│
├── services/
│   ├── motif-detection
│   ├── poly-optimizer-client
│   └── al-uls-orchestrator
│
├── apps/
│   ├── choppy-backend
│   └── choppy-frontend
│
└── infra/
    └── (orchestrator and deployment configs)
```

## Prerequisites

1. **GitLab Account**: Create an account at [gitlab.com](https://gitlab.com)
2. **GitHub Account**: Ensure you have access to your GitHub repositories
3. **Required Tools**: `git`, `curl`, `jq` (install with `apt-get install jq` on Ubuntu/Debian)

## Step 1: Create Access Tokens

### GitLab Personal Access Token

1. Go to [GitLab Personal Access Tokens](https://gitlab.com/-/profile/personal_access_tokens)
2. Create a new token with the following scopes:
   - `api`
   - `read_user`
   - `read_repository`
   - `write_repository`
3. Copy the token (you won't see it again)

### GitHub Personal Access Token (Optional but Recommended)

1. Go to [GitHub Personal Access Tokens](https://github.com/settings/tokens)
2. Create a new token with the `repo` scope
3. Copy the token

## Step 2: Configure Environment

1. Copy the example environment file:
   ```bash
   cp gitlab-migration.env.example gitlab-migration.env
   ```

2. Edit `gitlab-migration.env` with your values:
   ```bash
   nano gitlab-migration.env
   ```

3. Source the environment file:
   ```bash
   source gitlab-migration.env
   ```

## Step 3: Run Automated Migration

The migration script will automatically:

1. Create the GitLab group structure
2. Create all projects in the appropriate subgroups
3. Import repositories from GitHub
4. Enable container registries
5. Set up CI/CD variables

```bash
# Make the script executable
chmod +x gitlab-migration-setup.sh

# Run the migration
./gitlab-migration-setup.sh
```

## Step 4: Complete Manual Setup

After the automated migration, complete these steps:

### 1. Verify Repository Imports

For each project in GitLab:
1. Go to the project page
2. Navigate to **Settings > General > Advanced**
3. Check the import status
4. If imports failed, manually import using **Import project > Repo by URL**

### 2. Configure CI/CD Variables

1. Go to your `limps-suite` group
2. Navigate to **Settings > CI/CD > Variables**
3. Add the following variables:

| Variable | Value | Protected | Masked | Description |
|----------|-------|-----------|--------|-------------|
| `PYPI_TOKEN` | Your PyPI token | ✅ | ✅ | For Python package deployment |
| `STAGING_URL` | Your staging URL | ✅ | ❌ | Staging deployment URL |
| `PRODUCTION_URL` | Your production URL | ✅ | ❌ | Production deployment URL |

### 3. Register GitLab Runners

1. Go to your `limps-suite` group
2. Navigate to **CI/CD > Runners**
3. Register at least one Docker runner
4. For GPU workloads, consider registering a GPU-enabled runner

## Step 5: Deploy CI/CD Scaffold

Use the existing deployment script to copy CI/CD files to all repositories:

```bash
# Run the deployment script
./deploy-to-repos.sh

# Or specify a custom repository directory
./deploy-to-repos.sh /path/to/your/repos
```

This will:
- Copy `.gitlab-ci.yml` and `Dockerfile` to each project
- Create feature branches for the changes
- Prepare commits for all repositories

## Step 6: Push and Merge

### Push All Branches

```bash
# Push all branches to GitLab
for repo in ~/code/*; do
    if [ -d "$repo/.git" ]; then
        echo "Pushing $repo..."
        cd "$repo"
        git push origin add-gitlab-ci 2>/dev/null || git push origin add-orchestrator 2>/dev/null
        cd - > /dev/null
    fi
done
```

### Create Merge Requests

Create merge requests in this specific order:

1. **Core Services** (build first):
   - `limps-matrix-optimizer`
   - `symbolic-polynomial-svc`
   - `entropy-engine`

2. **Service Layer** (depend on core):
   - `motif-detection`
   - `poly-optimizer-client`
   - `al-uls-orchestrator`

3. **Applications** (depend on services):
   - `choppy-backend`
   - `choppy-frontend`

4. **Infrastructure** (orchestrates everything):
   - `infra` (merge last)

## Step 7: Verify Pipeline

After merging all projects:

1. Go to the `infra` project
2. Check the **CI/CD > Pipelines** page
3. The orchestrator pipeline should trigger automatically
4. Monitor the pipeline execution

## Troubleshooting

### Common Issues

1. **Import Failures**: 
   - Check GitHub token permissions
   - Verify repository URLs
   - Use manual import if needed

2. **Pipeline Failures**:
   - Check runner availability
   - Verify CI/CD variables are set
   - Review pipeline logs

3. **Permission Issues**:
   - Ensure GitLab token has correct scopes
   - Check group/project permissions

### Manual Repository Import

If automatic imports fail:

1. Go to the GitLab project
2. Click **Import project**
3. Select **Repo by URL**
4. Enter: `https://github.com/YOUR_USERNAME/REPO_NAME.git`
5. Provide GitHub credentials if prompted

## Advanced Configuration

### Custom GitLab Instance

To use a self-hosted GitLab instance, modify the script:

```bash
# Set custom GitLab URL
export GITLAB_URL="https://your-gitlab-instance.com"
```

### Custom Repository Mapping

Edit the repository mapping in `gitlab-migration-setup.sh`:

```bash
declare -A repos=(
    ["your-repo-name"]="subgroup/project-path|group_id|description"
    # Add more repositories here
)
```

## Support

If you encounter issues:

1. Check the script output for error messages
2. Verify all environment variables are set correctly
3. Ensure you have the required permissions
4. Review GitLab and GitHub API documentation

## Next Steps

After successful migration:

1. **Monitor Pipelines**: Set up notifications for pipeline status
2. **Optimize Builds**: Configure caching and parallel builds
3. **Security Scanning**: Enable security scanning in GitLab
4. **Documentation**: Update documentation with new GitLab URLs
5. **Team Access**: Invite team members to the GitLab group

---

**Note**: This migration preserves your Git history and maintains all your existing code. The CI/CD pipeline will be enhanced with GitLab's advanced features for better automation and deployment capabilities.