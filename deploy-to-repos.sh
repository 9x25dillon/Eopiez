#!/usr/bin/env bash
set -euo pipefail

# Configuration
REPO_BASE_DIR="${1:-~/code}"
EXTRACT_DIR="limps-suite-exec-final"
ZIP_FILE="limps-suite-exec-final.zip"

echo "=== limps-suite CI/CD Deployment Script ==="
echo "Repository base directory: $REPO_BASE_DIR"
echo ""

# Function to check if directory exists
check_repo() {
    local repo_path="$REPO_BASE_DIR/$1"
    if [[ ! -d "$repo_path" ]]; then
        echo "❌ Repository not found: $repo_path"
        echo "   Please ensure all repositories are cloned to $REPO_BASE_DIR"
        return 1
    fi
    echo "✅ Found repository: $1"
}

# Function to copy files and create commit
deploy_to_repo() {
    local repo_name="$1"
    local source_dir="$2"
    local files=("${@:3}")
    local repo_path="$REPO_BASE_DIR/$repo_name"
    
    echo ""
    echo "📦 Deploying to $repo_name..."
    
    # Check if repo exists
    if [[ ! -d "$repo_path" ]]; then
        echo "❌ Repository not found: $repo_path"
        return 1
    fi
    
    # Copy files
    for file in "${files[@]}"; do
        local source_file="$EXTRACT_DIR/$source_dir/$file"
        local dest_file="$repo_path/$file"
        
        if [[ -f "$source_file" ]]; then
            cp "$source_file" "$dest_file"
            echo "   ✅ Copied: $file"
        else
            echo "   ❌ Source file not found: $source_file"
            return 1
        fi
    done
    
    # Create git branch and commit
    cd "$repo_path"
    git checkout -b add-gitlab-ci 2>/dev/null || git checkout add-gitlab-ci
    
    # Add all copied files
    git add "${files[@]}"
    
    # Commit
    git commit -m "Add GitLab CI/CD pipeline and Dockerfile" || {
        echo "   ⚠️  No changes to commit (files may already exist)"
        return 0
    }
    
    echo "   ✅ Committed changes to branch 'add-gitlab-ci'"
    echo "   📤 Push with: git push origin add-gitlab-ci"
}

# Function to deploy infrastructure
deploy_infra() {
    local repo_name="limps-infra"
    local repo_path="$REPO_BASE_DIR/$repo_name"
    
    echo ""
    echo "🏗️  Deploying infrastructure to $repo_name..."
    
    if [[ ! -d "$repo_path" ]]; then
        echo "❌ Repository not found: $repo_path"
        return 1
    fi
    
    # Copy entire infra directory
    cp -r "$EXTRACT_DIR/infra/"* "$repo_path/"
    echo "   ✅ Copied infrastructure files"
    
    # Create git branch and commit
    cd "$repo_path"
    git checkout -b add-orchestrator 2>/dev/null || git checkout add-orchestrator
    
    # Add all files
    git add .
    
    # Commit
    git commit -m "Add orchestrator pipeline and deployment manifests" || {
        echo "   ⚠️  No changes to commit (files may already exist)"
        return 0
    }
    
    echo "   ✅ Committed changes to branch 'add-orchestrator'"
    echo "   📤 Push with: git push origin add-orchestrator"
}

# Main deployment process
main() {
    echo "🔍 Checking repository structure..."
    
    # Check all required repositories
    local repos=(
        "9xdSq-LIMPS-FemTO-R1C"
        "symbolic-polynomial-server"
        "entropy-engine"
        "LiMp"
        "motif-detection"
        "poly-optimizer-client"
        "Choppy-Backend"
        "Choppy-Frontend"
        "limps-infra"
    )
    
    for repo in "${repos[@]}"; do
        check_repo "$repo"
    done
    
    echo ""
    echo "🚀 Starting deployment process..."
    
    # Deploy to core repositories
    echo ""
    echo "=== Core Services ==="
    deploy_to_repo "9xdSq-LIMPS-FemTO-R1C" "core/limps-matrix-optimizer" ".gitlab-ci.yml" "Dockerfile"
    deploy_to_repo "symbolic-polynomial-server" "core/symbolic-polynomial-svc" ".gitlab-ci.yml" "Dockerfile"
    deploy_to_repo "entropy-engine" "core/entropy-engine" ".gitlab-ci.yml" "Dockerfile"
    
    # Deploy to service repositories
    echo ""
    echo "=== Service Layer ==="
    deploy_to_repo "LiMp" "services/al-uls-orchestrator" ".gitlab-ci.yml" "Dockerfile"
    deploy_to_repo "motif-detection" "services/motif-detection" ".gitlab-ci.yml" "Dockerfile"
    deploy_to_repo "poly-optimizer-client" "services/poly-optimizer-client" ".gitlab-ci.yml" "pyproject.toml"
    
    # Deploy to application repositories
    echo ""
    echo "=== Applications ==="
    deploy_to_repo "Choppy-Backend" "apps/choppy-backend" ".gitlab-ci.yml" "Dockerfile"
    deploy_to_repo "Choppy-Frontend" "apps/choppy-frontend" ".gitlab-ci.yml" "Dockerfile"
    
    # Deploy infrastructure
    deploy_infra
    
    echo ""
    echo "🎉 Deployment completed!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Push all branches to GitLab:"
    echo "   for repo in $REPO_BASE_DIR/*; do"
    echo "     cd \$repo && git push origin add-gitlab-ci"
    echo "   done"
    echo ""
    echo "2. Create merge requests in GitLab in this order:"
    echo "   - Core services (9xdSq-LIMPS-FemTO-R1C, symbolic-polynomial-server, entropy-engine)"
    echo "   - Service layer (LiMp, motif-detection, poly-optimizer-client)"
    echo "   - Applications (Choppy-Backend, Choppy-Frontend)"
    echo "   - Infrastructure (limps-infra)"
    echo ""
    echo "3. Configure GitLab variables:"
    echo "   - PYPI_TOKEN (for poly-optimizer-client)"
    echo "   - STAGING_URL (for deployment)"
    echo ""
    echo "4. Enable Container Registry and register runners"
    echo ""
    echo "5. Merge limps-infra last - it will trigger the orchestrator pipeline"
}

# Check if we're in the right directory
if [[ ! -f "$ZIP_FILE" ]]; then
    echo "❌ ZIP file not found: $ZIP_FILE"
    echo "   Please ensure you're in the directory containing the ZIP file"
    exit 1
fi

# Extract if needed
if [[ ! -d "$EXTRACT_DIR" ]]; then
    echo "📦 Extracting $ZIP_FILE..."
    unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"
    echo "✅ Extraction complete"
fi

# Run main deployment
main