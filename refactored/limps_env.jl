# --------------------------------------------------------------
# limps_env.jl - Refactored LIMPS Environment Configuration
#
# Load and validate environment variables for Julia code with improved
# error handling, type safety, and structured configuration.
#
# Usage:
#   using .LimpsEnv
#   println("API URL: ", LimpsEnv.config.api.url)
#   
#   # Legacy compatibility
#   println("API URL: ", LimpsEnv.CONFIG["LIMPS_API_URL"])
# --------------------------------------------------------------

module LimpsEnv

using Pkg
using Logging

# Try to load DotEnv if available
try
    using DotEnv
    global HAS_DOTENV = true
catch
    @warn "DotEnv.jl not found. Install with: Pkg.add(\"DotEnv\")"
    global HAS_DOTENV = false
end

# --------------------------------------------------------------
# Configuration Structures
# --------------------------------------------------------------

"""
    PathConfig

Core path configurations for LIMPS.
"""
struct PathConfig
    home::String
    julia_path::String
    python_path::String
    
    function PathConfig(home, julia_path, python_path)
        # Validate paths exist
        for (name, path) in [("home", home), ("julia_path", julia_path), ("python_path", python_path)]
            if !isdir(path)
                @warn "Directory does not exist: $name = $path"
            end
        end
        new(home, julia_path, python_path)
    end
end

"""
    LanguageConfig

Language environment configurations.
"""
struct LanguageConfig
    julia_project::String
    julia_load_path::String
    python_path::String
end

"""
    GPUConfig

GPU and CUDA configuration.
"""
struct GPUConfig
    cuda_visible_devices::String
    use_gpu::Bool
    
    function GPUConfig(cuda_devices, use_gpu)
        new(cuda_devices, use_gpu)
    end
end

# Helper function for GPU config
gpu_count(gpu::GPUConfig) = gpu.use_gpu ? length(split(gpu.cuda_visible_devices, ",")) : 0

"""
    APIConfig

API server configuration.
"""
struct APIConfig
    host::String
    port::Int
    
    function APIConfig(host, port)
        if !(1 <= port <= 65535)
            throw(ArgumentError("Invalid port number: $port"))
        end
        new(host, port)
    end
end

# Helper function for API config
api_url(api::APIConfig) = "http://$(api.host):$(api.port)"

"""
    DatabaseConfig

Database configuration.
"""
struct DatabaseConfig
    postgres_dsn::String
    redis_url::String
end

"""
    MonitoringConfig

Monitoring service configuration.
"""
struct MonitoringConfig
    prometheus_port::Int
    grafana_port::Int
    
    function MonitoringConfig(prom_port, grafana_port)
        if !(1 <= prom_port <= 65535)
            throw(ArgumentError("Invalid Prometheus port: $prom_port"))
        end
        if !(1 <= grafana_port <= 65535)
            throw(ArgumentError("Invalid Grafana port: $grafana_port"))
        end
        new(prom_port, grafana_port)
    end
end

"""
    LIMPSConfig

Complete LIMPS configuration.
"""
struct LIMPSConfig
    paths::PathConfig
    languages::LanguageConfig
    gpu::GPUConfig
    api::APIConfig
    database::DatabaseConfig
    monitoring::MonitoringConfig
end

# --------------------------------------------------------------
# Environment Loading
# --------------------------------------------------------------

"""
    load_dotenv()

Load .env.limps file if available and DotEnv.jl is installed.
"""
function load_dotenv()
    if !HAS_DOTENV
        return
    end
    
    # Look for .env.limps in current directory and parent directories
    current = pwd()
    for path in [current, dirname(current), dirname(dirname(current))]
        env_file = joinpath(path, ".env.limps")
        if isfile(env_file)
            DotEnv.load(env_file)
            @info "Loaded environment from: $env_file"
            break
        end
    end
end

"""
    require_env(key::String)

Get required environment variable or throw error.
"""
function require_env(key::String)
    value = get(ENV, key, "")
    if isempty(value)
        error("Required environment variable '$key' is not set. Please run 'source ./setup_limps_env.sh' first.")
    end
    return value
end

"""
    get_env(key::String, default::String="")

Get environment variable with default.
"""
function get_env(key::String, default::String="")
    return get(ENV, key, default)
end

"""
    get_env_bool(key::String, default::Bool=false)

Get boolean environment variable.
"""
function get_env_bool(key::String, default::Bool=false)
    value = lowercase(get_env(key, ""))
    if value in ["1", "true", "yes", "on"]
        return true
    elseif value in ["0", "false", "no", "off"]
        return false
    else
        return default
    end
end

"""
    get_env_int(key::String, default::Int)

Get integer environment variable with validation.
"""
function get_env_int(key::String, default::Int)
    value = get_env(key)
    if isempty(value)
        return default
    end
    
    try
        return parse(Int, value)
    catch
        @warn "Invalid integer value for $key: $value, using default $default"
        return default
    end
end

"""
    create_config()

Create configuration from environment variables.
"""
function create_config()
    load_dotenv()
    
    # Core paths
    limps_home = require_env("LIMPS_HOME")
    paths = PathConfig(
        limps_home,
        require_env("LIMPS_JULIA_PATH"),
        require_env("LIMPS_PYTHON_PATH")
    )
    
    # Language environments
    languages = LanguageConfig(
        require_env("JULIA_PROJECT"),
        get_env("JULIA_LOAD_PATH"),
        get_env("PYTHONPATH")
    )
    
    # GPU configuration
    gpu = GPUConfig(
        get_env("CUDA_VISIBLE_DEVICES", "0"),
        get_env_bool("USE_GPU", true)
    )
    
    # API configuration
    api = APIConfig(
        get_env("LIMPS_API_HOST", "localhost"),
        get_env_int("LIMPS_API_PORT", 8000)
    )
    
    # Database configuration
    database = DatabaseConfig(
        require_env("POSTGRES_DSN"),
        require_env("REDIS_URL")
    )
    
    # Monitoring configuration
    monitoring = MonitoringConfig(
        get_env_int("PROMETHEUS_PORT", 9090),
        get_env_int("GRAFANA_PORT", 3000)
    )
    
    return LIMPSConfig(paths, languages, gpu, api, database, monitoring)
end

# --------------------------------------------------------------
# Public Interface
# --------------------------------------------------------------

# Create the global configuration instance
try
    global config = create_config()
catch e
    @error "Failed to load LIMPS configuration" exception=(e, catch_backtrace())
    println(stderr, "❌ Failed to load LIMPS configuration: $e")
    println(stderr, "   Please run 'source ./setup_limps_env.sh' first.")
    exit(1)
end

"""
    get_config(key::String, default::Any=nothing)

Get configuration value by key path.
"""
function get_config(key::String, default::Any=nothing)
    try
        obj = config
        for part in split(key, ".")
            obj = getfield(obj, Symbol(part))
        end
        return obj
    catch
        return default
    end
end

"""
    validate_config()

Validate the current configuration.
"""
function validate_config()
    errors = String[]
    
    # Validate paths
    required_paths = [
        ("LIMPS_HOME", config.paths.home),
        ("LIMPS_JULIA_PATH", config.paths.julia_path),
        ("LIMPS_PYTHON_PATH", config.paths.python_path)
    ]
    
    for (name, path) in required_paths
        if !isdir(path)
            push!(errors, "Required path does not exist: $name = $path")
        end
    end
    
    # Validate URLs (basic format check)
    for (name, url) in [("PostgreSQL DSN", config.database.postgres_dsn),
                       ("Redis URL", config.database.redis_url)]
        if !startswith(url, "postgresql://") && !startswith(url, "redis://")
            push!(errors, "Invalid URL format for $name: $url")
        end
    end
    
    if !isempty(errors)
        error("Configuration validation failed:\n" * join("  • " .* errors, "\n"))
    end
end

"""
    print_config()

Print current configuration in a readable format.
"""
function print_config()
    println("🚀 LIMPS Configuration")
    println("=" ^ 50)
    println("Home Directory: $(config.paths.home)")
    println("API URL: $(api_url(config.api))")
    println("GPU Enabled: $(config.gpu.use_gpu)")
    println("GPU Devices: $(config.gpu.cuda_visible_devices)")
    println("PostgreSQL: $(config.database.postgres_dsn)")
    println("Redis: $(config.database.redis_url)")
    println("=" ^ 50)
end

# --------------------------------------------------------------
# Backward Compatibility
# --------------------------------------------------------------

# Legacy dictionary interface for backward compatibility
const CONFIG = Dict{String, Any}(
    "LIMPS_HOME" => config.paths.home,
    "LIMPS_JULIA_PATH" => config.paths.julia_path,
    "LIMPS_PYTHON_PATH" => config.paths.python_path,
    "JULIA_PROJECT" => config.languages.julia_project,
    "JULIA_LOAD_PATH" => config.languages.julia_load_path,
    "PYTHONPATH" => config.languages.python_path,
    "CUDA_VISIBLE_DEVICES" => config.gpu.cuda_visible_devices,
    "USE_GPU" => config.gpu.use_gpu,
    "LIMPS_API_HOST" => config.api.host,
    "LIMPS_API_PORT" => config.api.port,
    "LIMPS_API_URL" => api_url(config.api),
    "POSTGRES_DSN" => config.database.postgres_dsn,
    "REDIS_URL" => config.database.redis_url,
    "PROMETHEUS_PORT" => config.monitoring.prometheus_port,
    "GRAFANA_PORT" => config.monitoring.grafana_port,
)

# Legacy get function
function get(key::String, default::Any=nothing)
    return get(CONFIG, key, default)
end

# --------------------------------------------------------------
# Module Testing
# --------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    println("Testing LIMPS configuration...")
    try
        validate_config()
        print_config()
        println("✅ Configuration is valid!")
    catch e
        println("❌ Configuration error: $e")
        exit(1)
    end
end

end # module