#!/usr/bin/env bash
set -euo pipefail

# GitLab Migration Setup Script
# This script helps set up the GitLab group structure and migrate repositories from GitHub

# Configuration
GITLAB_GROUP="limps-suite"
GITHUB_USERNAME="${GITHUB_USERNAME:-}"
GITLAB_TOKEN="${GITLAB_TOKEN:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check required tools
check_requirements() {
    print_status "Checking required tools..."
    
    local missing_tools=()
    
    if ! command_exists git; then
        missing_tools+=("git")
    fi
    
    if ! command_exists curl; then
        missing_tools+=("curl")
    fi
    
    if ! command_exists jq; then
        missing_tools+=("jq")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_status "Please install them and try again"
        exit 1
    fi
    
    print_success "All required tools are available"
}

# Function to validate configuration
validate_config() {
    print_status "Validating configuration..."
    
    if [[ -z "$GITHUB_USERNAME" ]]; then
        print_error "GITHUB_USERNAME is not set"
        print_status "Please set it: export GITHUB_USERNAME='your-github-username'"
        exit 1
    fi
    
    if [[ -z "$GITLAB_TOKEN" ]]; then
        print_error "GITLAB_TOKEN is not set"
        print_status "Please set it: export GITLAB_TOKEN='your-gitlab-token'"
        exit 1
    fi
    
    if [[ -z "$GITHUB_TOKEN" ]]; then
        print_warning "GITHUB_TOKEN is not set - some operations may fail"
        print_status "Please set it: export GITHUB_TOKEN='your-github-token'"
    fi
    
    print_success "Configuration validated"
}

# Function to create GitLab group
create_gitlab_group() {
    print_status "Creating GitLab group: $GITLAB_GROUP"
    
    local response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        -d "{
            \"name\": \"$GITLAB_GROUP\",
            \"path\": \"$GITLAB_GROUP\",
            \"description\": \"LIMPS Suite - Machine Learning and Optimization Platform\",
            \"visibility\": \"private\"
        }" \
        "https://gitlab.com/api/v4/groups")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Group created successfully"
        echo "$response" | jq -r '.id'
    else
        if echo "$response" | jq -e '.message' | grep -q "has already been taken"; then
            print_warning "Group already exists, getting group ID..."
            get_gitlab_group_id
        else
            print_error "Failed to create group: $(echo "$response" | jq -r '.message // .error // "Unknown error"')"
            exit 1
        fi
    fi
}

# Function to get GitLab group ID
get_gitlab_group_id() {
    local response
    response=$(curl -s -X GET \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        "https://gitlab.com/api/v4/groups/$GITLAB_GROUP")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Found existing group"
        echo "$response" | jq -r '.id'
    else
        print_error "Failed to get group ID"
        exit 1
    fi
}

# Function to create subgroup
create_subgroup() {
    local parent_group_id="$1"
    local subgroup_name="$2"
    local subgroup_path="$3"
    local description="$4"
    
    print_status "Creating subgroup: $subgroup_name"
    
    local response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        -d "{
            \"name\": \"$subgroup_name\",
            \"path\": \"$subgroup_path\",
            \"parent_id\": $parent_group_id,
            \"description\": \"$description\",
            \"visibility\": \"private\"
        }" \
        "https://gitlab.com/api/v4/groups")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Subgroup '$subgroup_name' created successfully"
        echo "$response" | jq -r '.id'
    else
        if echo "$response" | jq -e '.message' | grep -q "has already been taken"; then
            print_warning "Subgroup '$subgroup_name' already exists, getting ID..."
            get_subgroup_id "$parent_group_id" "$subgroup_path"
        else
            print_error "Failed to create subgroup '$subgroup_name': $(echo "$response" | jq -r '.message // .error // "Unknown error"')"
            exit 1
        fi
    fi
}

# Function to get subgroup ID
get_subgroup_id() {
    local parent_group_id="$1"
    local subgroup_path="$2"
    
    local response
    response=$(curl -s -X GET \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        "https://gitlab.com/api/v4/groups/$GITLAB_GROUP%2F$subgroup_path")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Found existing subgroup"
        echo "$response" | jq -r '.id'
    else
        print_error "Failed to get subgroup ID"
        exit 1
    fi
}

# Function to create project
create_project() {
    local group_id="$1"
    local project_name="$2"
    local project_path="$3"
    local description="$4"
    
    print_status "Creating project: $project_name"
    
    local response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        -d "{
            \"name\": \"$project_name\",
            \"path\": \"$project_path\",
            \"namespace_id\": $group_id,
            \"description\": \"$description\",
            \"visibility\": \"private\",
            \"initialize_with_readme\": false
        }" \
        "https://gitlab.com/api/v4/projects")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Project '$project_name' created successfully"
        echo "$response" | jq -r '.id'
    else
        if echo "$response" | jq -e '.message' | grep -q "has already been taken"; then
            print_warning "Project '$project_name' already exists, getting ID..."
            get_project_id "$group_id" "$project_path"
        else
            print_error "Failed to create project '$project_name': $(echo "$response" | jq -r '.message // .error // "Unknown error"')"
            exit 1
        fi
    fi
}

# Function to get project ID
get_project_id() {
    local group_id="$1"
    local project_path="$2"
    
    local response
    response=$(curl -s -X GET \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        "https://gitlab.com/api/v4/projects/$GITLAB_GROUP%2F$project_path")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Found existing project"
        echo "$response" | jq -r '.id'
    else
        print_error "Failed to get project ID"
        exit 1
    fi
}

# Function to import repository from GitHub
import_repository() {
    local project_id="$1"
    local github_repo="$2"
    
    print_status "Importing repository: $github_repo"
    
    local import_url="https://github.com/$github_repo.git"
    local auth_url=""
    
    if [[ -n "$GITHUB_TOKEN" ]]; then
        auth_url="https://$GITHUB_TOKEN@github.com/$github_repo.git"
    else
        auth_url="$import_url"
    fi
    
    local response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        -d "{
            \"repo_url\": \"$auth_url\",
            \"target_namespace\": \"$GITLAB_GROUP\"
        }" \
        "https://gitlab.com/api/v4/projects/$project_id/import")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Import started for $github_repo"
        echo "$response" | jq -r '.id'
    else
        print_error "Failed to start import: $(echo "$response" | jq -r '.message // .error // "Unknown error"')"
        return 1
    fi
}

# Function to set up CI/CD variables
setup_cicd_variables() {
    local group_id="$1"
    
    print_status "Setting up CI/CD variables for group..."
    
    # List of variables to set up
    local variables=(
        '{"key": "PYPI_TOKEN", "value": "", "protected": true, "masked": true, "description": "PyPI token for package deployment"}'
        '{"key": "STAGING_URL", "value": "", "protected": true, "masked": false, "description": "Staging deployment URL"}'
        '{"key": "PRODUCTION_URL", "value": "", "protected": true, "masked": false, "description": "Production deployment URL"}'
    )
    
    for var_json in "${variables[@]}"; do
        local key=$(echo "$var_json" | jq -r '.key')
        print_status "Setting up variable: $key"
        
        local response
        response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $GITLAB_TOKEN" \
            -d "$var_json" \
            "https://gitlab.com/api/v4/groups/$group_id/variables")
        
        if echo "$response" | jq -e '.key' >/dev/null 2>&1; then
            print_success "Variable '$key' created successfully"
        else
            if echo "$response" | jq -e '.message' | grep -q "has already been taken"; then
                print_warning "Variable '$key' already exists"
            else
                print_error "Failed to create variable '$key': $(echo "$response" | jq -r '.message // .error // "Unknown error"')"
            fi
        fi
    done
}

# Function to enable container registry
enable_container_registry() {
    local project_id="$1"
    local project_name="$2"
    
    print_status "Enabling container registry for: $project_name"
    
    local response
    response=$(curl -s -X PUT \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GITLAB_TOKEN" \
        -d '{"container_registry_enabled": true}' \
        "https://gitlab.com/api/v4/projects/$project_id")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        print_success "Container registry enabled for $project_name"
    else
        print_error "Failed to enable container registry: $(echo "$response" | jq -r '.message // .error // "Unknown error"')"
    fi
}

# Main function
main() {
    echo "=== GitLab Migration Setup Script ==="
    echo "This script will help you set up your GitLab group structure and migrate repositories"
    echo ""
    
    check_requirements
    validate_config
    
    print_status "Starting GitLab group setup..."
    
    # Create main group
    local main_group_id
    main_group_id=$(create_gitlab_group)
    print_success "Main group ID: $main_group_id"
    
    # Create subgroups
    print_status "Creating subgroups..."
    
    local core_group_id
    core_group_id=$(create_subgroup "$main_group_id" "core" "core" "Core mathematical and optimization libraries")
    
    local services_group_id
    services_group_id=$(create_subgroup "$main_group_id" "services" "services" "Microservices and API layer")
    
    local apps_group_id
    apps_group_id=$(create_subgroup "$main_group_id" "apps" "apps" "End-user applications")
    
    local infra_group_id
    infra_group_id=$(create_subgroup "$main_group_id" "infra" "infra" "Infrastructure and deployment configuration")
    
    print_success "All subgroups created successfully"
    
    # Define repository mapping
    declare -A repos=(
        # Core repositories
        ["limps-matrix-optimizer"]="core/limps-matrix-optimizer|$core_group_id|Matrix optimization algorithms for LIMPS"
        ["symbolic-polynomial-svc"]="core/symbolic-polynomial-svc|$core_group_id|Symbolic polynomial support vector classifier"
        ["entropy-engine"]="core/entropy-engine|$core_group_id|Entropy calculation and analysis engine"
        
        # Service repositories
        ["motif-detection"]="services/motif-detection|$services_group_id|Pattern and motif detection service"
        ["poly-optimizer-client"]="services/poly-optimizer-client|$services_group_id|Polynomial optimizer client library"
        ["al-uls-orchestrator"]="services/al-uls-orchestrator|$services_group_id|Active learning ULS orchestrator"
        
        # Application repositories
        ["choppy-backend"]="apps/choppy-backend|$apps_group_id|Choppy application backend API"
        ["choppy-frontend"]="apps/choppy-frontend|$apps_group_id|Choppy application frontend UI"
        
        # Infrastructure
        ["infra"]="infra|$infra_group_id|Infrastructure and deployment configuration"
    )
    
    # Create projects and import repositories
    print_status "Creating projects and importing repositories..."
    
    for repo_name in "${!repos[@]}"; do
        IFS='|' read -r project_path group_id description <<< "${repos[$repo_name]}"
        
        # Extract just the project name from the path
        local project_name=$(basename "$project_path")
        
        # Create project
        local project_id
        project_id=$(create_project "$group_id" "$project_name" "$project_name" "$description")
        
        # Enable container registry (except for infra)
        if [[ "$repo_name" != "infra" ]]; then
            enable_container_registry "$project_id" "$project_name"
        fi
        
        # Import from GitHub if not infra
        if [[ "$repo_name" != "infra" ]]; then
            local github_repo="$GITHUB_USERNAME/$repo_name"
            import_repository "$project_id" "$github_repo"
        fi
        
        echo ""
    done
    
    # Set up CI/CD variables
    setup_cicd_variables "$main_group_id"
    
    echo ""
    echo "=== Migration Setup Complete! ==="
    echo ""
    echo "📋 Next Steps:"
    echo ""
    echo "1. Complete GitHub repository imports:"
    echo "   - Go to each project in GitLab"
    echo "   - Check the import status in Settings > General > Advanced"
    echo "   - If imports failed, manually import using 'Repo by URL'"
    echo ""
    echo "2. Set up your CI/CD variables:"
    echo "   - Go to $GITLAB_GROUP group > Settings > CI/CD > Variables"
    echo "   - Add your actual values for PYPI_TOKEN, STAGING_URL, etc."
    echo ""
    echo "3. Register GitLab Runners:"
    echo "   - Go to $GITLAB_GROUP group > CI/CD > Runners"
    echo "   - Register Docker runners (or GPU runners if needed)"
    echo ""
    echo "4. Deploy your CI/CD scaffold:"
    echo "   - Run: ./deploy-to-repos.sh"
    echo "   - This will copy .gitlab-ci.yml and Dockerfile to each project"
    echo ""
    echo "5. Push and merge:"
    echo "   - Push all branches to GitLab"
    echo "   - Create merge requests in the order specified in deploy-to-repos.sh"
    echo ""
    echo "6. Final orchestration:"
    echo "   - Merge the infra project last"
    echo "   - This will trigger the orchestrator pipeline"
    echo ""
    echo "🌐 Your GitLab group structure:"
    echo "   https://gitlab.com/$GITLAB_GROUP"
    echo ""
}

# Run main function
main "$@"