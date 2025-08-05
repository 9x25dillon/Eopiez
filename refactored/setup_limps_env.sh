#!/usr/bin/env bash
# ----------------------------------------------------------------------
# setup_limps_env.sh - Refactored LIMPS Environment Setup
# 
# A self-contained Bash script that sets environment variables,
# validates paths, and generates configuration files for the LIMPS project.
# 
# Usage:
#   source ./setup_limps_env.sh    # Sets vars in current shell
#   ./setup_limps_env.sh           # Runs in subshell (for validation only)
# ----------------------------------------------------------------------

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ----------------------------------------------------------------------
# Configuration Constants
# ----------------------------------------------------------------------
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
readonly ENV_FILE="${REPO_ROOT}/.env.limps"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# ----------------------------------------------------------------------
# Logging Functions
# ----------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}ℹ️  $*${NC}" >&2
}

log_success() {
    echo -e "${GREEN}✅ $*${NC}" >&2
}

log_warning() {
    echo -e "${YELLOW}⚠️  $*${NC}" >&2
}

log_error() {
    echo -e "${RED}❌ $*${NC}" >&2
}

# ----------------------------------------------------------------------
# Environment Variable Definitions
# ----------------------------------------------------------------------
setup_core_paths() {
    export LIMPS_HOME="${LIMPS_HOME:-$REPO_ROOT}"
    export LIMPS_JULIA_PATH="${LIMPS_JULIA_PATH:-${LIMPS_HOME}/limps_core/julia}"
    export LIMPS_PYTHON_PATH="${LIMPS_PYTHON_PATH:-${LIMPS_HOME}/limps_core/python}"
}

setup_language_environments() {
    # Julia configuration
    export JULIA_PROJECT="${JULIA_PROJECT:-$LIMPS_JULIA_PATH}"
    export JULIA_LOAD_PATH="${JULIA_LOAD_PATH:-$LIMPS_JULIA_PATH}"
    
    # Python configuration
    export PYTHONPATH="${PYTHONPATH:-$LIMPS_HOME}"
}

setup_gpu_configuration() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export USE_GPU="${USE_GPU:-1}"
}

setup_api_configuration() {
    export LIMPS_API_HOST="${LIMPS_API_HOST:-localhost}"
    export LIMPS_API_PORT="${LIMPS_API_PORT:-8000}"
    export LIMPS_API_URL="http://${LIMPS_API_HOST}:${LIMPS_API_PORT}"
}

setup_database_configuration() {
    export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
    export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    export POSTGRES_USER="${POSTGRES_USER:-user}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-pass}"
    export POSTGRES_DB="${POSTGRES_DB:-limps_db}"
    export POSTGRES_DSN="${POSTGRES_DSN:-postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}}"
    
    export REDIS_HOST="${REDIS_HOST:-localhost}"
    export REDIS_PORT="${REDIS_PORT:-6379}"
    export REDIS_DB="${REDIS_DB:-0}"
    export REDIS_URL="${REDIS_URL:-redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}}"
}

setup_monitoring_configuration() {
    export PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
    export GRAFANA_PORT="${GRAFANA_PORT:-3000}"
}

# ----------------------------------------------------------------------
# Validation Functions
# ----------------------------------------------------------------------
declare -a VALIDATION_ERRORS=()

validate_directory() {
    local var_name="$1"
    local dir_path="$2"
    local is_required="${3:-true}"
    
    if [[ ! -d "$dir_path" ]]; then
        if [[ "$is_required" == "true" ]]; then
            VALIDATION_ERRORS+=("Required directory missing: $var_name ($dir_path)")
        else
            log_warning "Optional directory missing: $var_name ($dir_path)"
        fi
    fi
}

validate_port() {
    local var_name="$1"
    local port="$2"
    
    if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
        VALIDATION_ERRORS+=("Invalid port number: $var_name ($port)")
    fi
}

validate_environment() {
    log_info "Validating environment configuration..."
    
    # Validate required directories
    validate_directory "LIMPS_HOME" "$LIMPS_HOME"
    validate_directory "LIMPS_JULIA_PATH" "$LIMPS_JULIA_PATH"
    validate_directory "LIMPS_PYTHON_PATH" "$LIMPS_PYTHON_PATH"
    
    # Validate ports
    validate_port "LIMPS_API_PORT" "$LIMPS_API_PORT"
    validate_port "POSTGRES_PORT" "$POSTGRES_PORT"
    validate_port "REDIS_PORT" "$REDIS_PORT"
    validate_port "PROMETHEUS_PORT" "$PROMETHEUS_PORT"
    validate_port "GRAFANA_PORT" "$GRAFANA_PORT"
    
    # Check for required tools
    if ! command -v git >/dev/null 2>&1; then
        VALIDATION_ERRORS+=("Git is required but not installed")
    fi
    
    # Validate GPU configuration
    if [[ "$USE_GPU" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
        log_warning "USE_GPU=1 but nvidia-smi not found. GPU features may not work."
    fi
}

# ----------------------------------------------------------------------
# Configuration File Generation
# ----------------------------------------------------------------------
generate_env_file() {
    log_info "Generating .env.limps file..."
    
    cat > "$ENV_FILE" <<EOF
# --------------------------------------------------------------
# LIMPS unified .env file
# Generated by setup_limps_env.sh on $(date)
# --------------------------------------------------------------

# Core Paths
LIMPS_HOME=${LIMPS_HOME}
LIMPS_JULIA_PATH=${LIMPS_JULIA_PATH}
LIMPS_PYTHON_PATH=${LIMPS_PYTHON_PATH}

# Language Environments
JULIA_PROJECT=${JULIA_PROJECT}
JULIA_LOAD_PATH=${JULIA_LOAD_PATH}
PYTHONPATH=${PYTHONPATH}

# GPU Configuration
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
USE_GPU=${USE_GPU}

# API Configuration
LIMPS_API_HOST=${LIMPS_API_HOST}
LIMPS_API_PORT=${LIMPS_API_PORT}
LIMPS_API_URL=${LIMPS_API_URL}

# Database Configuration
POSTGRES_HOST=${POSTGRES_HOST}
POSTGRES_PORT=${POSTGRES_PORT}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_DSN=${POSTGRES_DSN}

# Redis Configuration
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
REDIS_DB=${REDIS_DB}
REDIS_URL=${REDIS_URL}

# Monitoring
PROMETHEUS_PORT=${PROMETHEUS_PORT}
GRAFANA_PORT=${GRAFANA_PORT}
EOF
    
    log_success ".env.limps written to ${ENV_FILE}"
}

# ----------------------------------------------------------------------
# Status Reporting
# ----------------------------------------------------------------------
print_environment_status() {
    echo
    echo "=================================================================="
    echo "🚀 LIMPS Environment Configuration"
    echo "=================================================================="
    echo
    
    printf "%-25s = %s\n" "Repository Root" "$LIMPS_HOME"
    echo
    
    echo "📁 Core Paths:"
    printf "%-25s = %s\n" "  Julia Path" "$LIMPS_JULIA_PATH"
    printf "%-25s = %s\n" "  Python Path" "$LIMPS_PYTHON_PATH"
    echo
    
    echo "🐍 Language Environments:"
    printf "%-25s = %s\n" "  Julia Project" "$JULIA_PROJECT"
    printf "%-25s = %s\n" "  Julia Load Path" "$JULIA_LOAD_PATH"
    printf "%-25s = %s\n" "  Python Path" "$PYTHONPATH"
    echo
    
    echo "🎮 GPU Configuration:"
    printf "%-25s = %s\n" "  CUDA Devices" "$CUDA_VISIBLE_DEVICES"
    printf "%-25s = %s\n" "  GPU Enabled" "$USE_GPU"
    echo
    
    echo "🌐 API Configuration:"
    printf "%-25s = %s\n" "  Host" "$LIMPS_API_HOST"
    printf "%-25s = %s\n" "  Port" "$LIMPS_API_PORT"
    printf "%-25s = %s\n" "  URL" "$LIMPS_API_URL"
    echo
    
    echo "🗄️ Database Configuration:"
    printf "%-25s = %s\n" "  PostgreSQL DSN" "$POSTGRES_DSN"
    printf "%-25s = %s\n" "  Redis URL" "$REDIS_URL"
    echo
    
    echo "📊 Monitoring:"
    printf "%-25s = %s\n" "  Prometheus Port" "$PROMETHEUS_PORT"
    printf "%-25s = %s\n" "  Grafana Port" "$GRAFANA_PORT"
    echo
    echo "=================================================================="
}

report_validation_results() {
    if (( ${#VALIDATION_ERRORS[@]} > 0 )); then
        echo
        log_error "Validation failed with ${#VALIDATION_ERRORS[@]} error(s):"
        for error in "${VALIDATION_ERRORS[@]}"; do
            echo "   • $error"
        done
        echo
        log_error "Please fix the above issues before proceeding."
        return 1
    else
        echo
        log_success "All validations passed successfully!"
        echo
        log_info "Next steps:"
        echo "   • Run: python -c \"from limps_env import CONFIG; print('✓ Python config loaded')\""
        echo "   • Run: julia -e 'include(\"julia/limps_env.jl\"); println(\"✓ Julia config loaded\")'"
        echo "   • Start services: docker compose up -d"
        return 0
    fi
}

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
main() {
    log_info "Setting up LIMPS environment..."
    
    # Set up all environment variables
    setup_core_paths
    setup_language_environments
    setup_gpu_configuration
    setup_api_configuration
    setup_database_configuration
    setup_monitoring_configuration
    
    # Validate the configuration
    validate_environment
    
    # Generate configuration files
    generate_env_file
    
    # Create symlink for IDE compatibility
    if [[ ! -L "${REPO_ROOT}/.env" ]]; then
        ln -sf .env.limps "${REPO_ROOT}/.env"
        log_info "Created .env symlink for IDE compatibility"
    fi
    
    # Print status report
    print_environment_status
    
    # Report validation results
    report_validation_results
}

# ----------------------------------------------------------------------
# Execute main function if script is sourced or run directly
# ----------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    main
else
    # Script is being sourced
    main
fi

# Export function for external use
export -f log_info log_success log_warning log_error