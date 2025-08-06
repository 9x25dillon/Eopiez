#!/usr/bin/env bash
set -euo pipefail

# Configuration
REPO_BASE_DIR="${1:-~/code}"

echo "=== Push All Branches to GitLab ==="
echo "Repository base directory: $REPO_BASE_DIR"
echo ""

# Function to push branch for a repository
push_repo() {
    local repo_path="$1"
    local repo_name=$(basename "$repo_path")
    
    echo "📤 Pushing $repo_name..."
    
    if [[ ! -d "$repo_path" ]]; then
        echo "   ❌ Repository not found: $repo_path"
        return 1
    fi
    
    cd "$repo_path"
    
    # Check which branch exists and push it
    if git branch | grep -q "add-gitlab-ci"; then
        echo "   🔄 Pushing add-gitlab-ci branch..."
        git push origin add-gitlab-ci
        echo "   ✅ Pushed add-gitlab-ci branch"
    elif git branch | grep -q "add-orchestrator"; then
        echo "   🔄 Pushing add-orchestrator branch..."
        git push origin add-orchestrator
        echo "   ✅ Pushed add-orchestrator branch"
    else
        echo "   ⚠️  No deployment branch found"
        return 1
    fi
}

# Main execution
main() {
    echo "🔍 Scanning repositories in $REPO_BASE_DIR..."
    
    # Expected repositories
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
    
    local success_count=0
    local total_count=${#repos[@]}
    
    for repo in "${repos[@]}"; do
        local repo_path="$REPO_BASE_DIR/$repo"
        if push_repo "$repo_path"; then
            ((success_count++))
        fi
        echo ""
    done
    
    echo "🎉 Push completed!"
    echo "✅ Successfully pushed: $success_count/$total_count repositories"
    
    if [[ $success_count -eq $total_count ]]; then
        echo ""
        echo "📋 Next Steps:"
        echo "1. Go to GitLab and create merge requests for each repository"
        echo "2. Merge in dependency order:"
        echo "   - Core services first"
        echo "   - Service layer second"
        echo "   - Applications third"
        echo "   - Infrastructure last"
        echo ""
        echo "3. Configure GitLab variables:"
        echo "   - PYPI_TOKEN"
        echo "   - STAGING_URL"
        echo ""
        echo "4. Enable Container Registry and register runners"
    else
        echo ""
        echo "⚠️  Some repositories failed to push. Please check:"
        echo "   - Repository exists in $REPO_BASE_DIR"
        echo "   - Deployment script was run successfully"
        echo "   - Git remote is configured correctly"
    fi
}

# Check if repository directory exists
if [[ ! -d "$REPO_BASE_DIR" ]]; then
    echo "❌ Repository directory not found: $REPO_BASE_DIR"
    echo "   Please specify the correct path to your repositories"
    echo "   Usage: $0 <repository-directory>"
    exit 1
fi

# Run main function
main