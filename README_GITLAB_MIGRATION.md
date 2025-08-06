# GitLab Migration for LIMPS Suite

This directory contains everything you need to migrate your GitHub repositories to GitLab and set up a comprehensive CI/CD pipeline for your LIMPS suite.

## 🚀 Quick Start

For the fastest setup, run the automated migration script:

```bash
./quick-migration-setup.sh
```

This will:
1. Check system requirements and install missing tools
2. Prompt for your configuration (GitHub username, tokens, etc.)
3. Create the GitLab group structure
4. Import all repositories from GitHub
5. Deploy the CI/CD scaffold
6. Guide you through the final steps

## 📁 Files Overview

### Migration Scripts

- **`quick-migration-setup.sh`** - Complete automated migration (recommended)
- **`gitlab-migration-setup.sh`** - Core migration script (API-based)
- **`deploy-to-repos.sh`** - Deploy CI/CD files to repositories (existing)

### Configuration

- **`gitlab-migration.env.example`** - Example environment configuration
- **`gitlab-migration.env`** - Your actual configuration (created by setup)

### Documentation

- **`GITLAB_MIGRATION_GUIDE.md`** - Comprehensive step-by-step guide
- **`README_GITLAB_MIGRATION.md`** - This file

## 🔧 Manual Setup

If you prefer manual control, follow these steps:

### 1. Configure Environment

```bash
# Copy and edit the example configuration
cp gitlab-migration.env.example gitlab-migration.env
nano gitlab-migration.env

# Source the environment
source gitlab-migration.env
```

### 2. Run Migration

```bash
# Make scripts executable
chmod +x gitlab-migration-setup.sh

# Run the migration
./gitlab-migration-setup.sh
```

### 3. Deploy CI/CD Scaffold

```bash
# Deploy to all repositories
./deploy-to-repos.sh

# Or specify a custom directory
./deploy-to-repos.sh /path/to/your/repos
```

## 🏗️ GitLab Structure

The migration creates this organized structure:

```
limps-suite/
│
├── core/                    # Core mathematical libraries
│   ├── limps-matrix-optimizer
│   ├── symbolic-polynomial-svc
│   └── entropy-engine
│
├── services/               # Microservices and APIs
│   ├── motif-detection
│   ├── poly-optimizer-client
│   └── al-uls-orchestrator
│
├── apps/                   # End-user applications
│   ├── choppy-backend
│   └── choppy-frontend
│
└── infra/                  # Infrastructure and deployment
    └── (orchestrator configs)
```

## 🔑 Required Tokens

### GitLab Personal Access Token
- **Scopes**: `api`, `read_user`, `read_repository`, `write_repository`
- **Create at**: https://gitlab.com/-/profile/personal_access_tokens

### GitHub Personal Access Token (Optional)
- **Scopes**: `repo`
- **Create at**: https://github.com/settings/tokens
- **Purpose**: For importing private repositories

## 📋 Post-Migration Steps

After running the migration scripts:

1. **Verify Imports**: Check each project's import status in GitLab
2. **Configure Variables**: Set up CI/CD variables (PYPI_TOKEN, STAGING_URL, etc.)
3. **Register Runners**: Set up GitLab runners for CI/CD execution
4. **Push Branches**: Push the generated feature branches to GitLab
5. **Create MRs**: Create merge requests in the specified order
6. **Monitor Pipelines**: Watch the orchestrator pipeline execute

## 🛠️ Advanced Usage

### Partial Migration

```bash
# Only create environment file
./quick-migration-setup.sh --setup-only

# Only run migration (requires env file)
./quick-migration-setup.sh --migrate-only

# Only deploy scaffold (requires env file)
./quick-migration-setup.sh --deploy-only
```

### Custom Configuration

Edit the repository mapping in `gitlab-migration-setup.sh`:

```bash
declare -A repos=(
    ["your-repo-name"]="subgroup/project-path|group_id|description"
    # Add more repositories here
)
```

### Self-Hosted GitLab

To use a self-hosted GitLab instance:

```bash
export GITLAB_URL="https://your-gitlab-instance.com"
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Failures**
   - Check GitHub token permissions
   - Verify repository URLs
   - Use manual import if needed

2. **Permission Errors**
   - Ensure GitLab token has correct scopes
   - Check group/project permissions

3. **Missing Tools**
   - The script will attempt to install missing tools
   - Manual installation may be required on some systems

### Manual Repository Import

If automatic imports fail:

1. Go to the GitLab project
2. Click **Import project**
3. Select **Repo by URL**
4. Enter: `https://github.com/YOUR_USERNAME/REPO_NAME.git`

## 📚 Additional Resources

- **GitLab API Documentation**: https://docs.gitlab.com/ee/api/
- **GitLab CI/CD Documentation**: https://docs.gitlab.com/ee/ci/
- **GitLab Runner Documentation**: https://docs.gitlab.com/runner/

## 🤝 Support

If you encounter issues:

1. Check the script output for error messages
2. Verify all environment variables are set correctly
3. Ensure you have the required permissions
4. Review the comprehensive guide in `GITLAB_MIGRATION_GUIDE.md`

---

**Note**: This migration preserves your Git history and maintains all your existing code. The CI/CD pipeline will be enhanced with GitLab's advanced features for better automation and deployment capabilities.