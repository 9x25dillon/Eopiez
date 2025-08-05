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
    database_name::String
    
    function DatabaseConfig(postgres_dsn, redis_url, database_name)
        # Basic validation
        if isempty(postgres_dsn)
            @warn "PostgreSQL DSN is empty"
        end
        if isempty(redis_url)
            @warn "Redis URL is empty"
        end
        new(postgres_dsn, redis_url, database_name)
    end
end

"""
    LoggingConfig

Logging configuration.
"""
struct LoggingConfig
    level::String
    file::String
    console::Bool
    
    function LoggingConfig(level, file, console)
        valid_levels = ["DEBUG", "INFO", "WARN", "ERROR"]
        if !(uppercase(level) in valid_levels)
            @warn "Invalid log level: $level. Using INFO instead."
            level = "INFO"
        end
        new(uppercase(level), file, console)
    end
end

"""
    SecurityConfig

Security and authentication configuration.
"""
struct SecurityConfig
    secret_key::String
    token_expiry::Int
    enable_ssl::Bool
    
    function SecurityConfig(secret_key, token_expiry, enable_ssl)
        if isempty(secret_key)
            @warn "Secret key is empty - this may cause security issues"
        end
        if token_expiry <= 0
            @warn "Token expiry must be positive, using default 3600"
            token_expiry = 3600
        end
        new(secret_key, token_expiry, enable_ssl)
    end
end

"""
    MainConfig

Main configuration structure containing all sub-configurations.
"""
struct MainConfig
    paths::PathConfig
    language::LanguageConfig
    gpu::GPUConfig
    api::APIConfig
    database::DatabaseConfig
    logging::LoggingConfig
    security::SecurityConfig
end

# --------------------------------------------------------------
# Environment Variable Loading
# --------------------------------------------------------------

"""
    load_env_file(file_path::String)

Load environment variables from a .env file if DotEnv is available.
"""
function load_env_file(file_path::String)
    if HAS_DOTENV && isfile(file_path)
        try
            DotEnv.load(file_path)
            @info "Loaded environment from: $file_path"
        catch e
            @warn "Failed to load .env file: $e"
        end
    end
end

"""
    get_env_var(key::String, default::String="")

Get environment variable with fallback to default value.
"""
function get_env_var(key::String, default::String="")
    value = get(ENV, key, default)
    if isempty(value) && !isempty(default)
        @warn "Environment variable $key is empty, using default: $default"
    end
    return value
end

"""
    get_env_var_int(key::String, default::Int)

Get environment variable as integer with fallback to default value.
"""
function get_env_var_int(key::String, default::Int)
    value = get_env_var(key, string(default))
    try
        return parse(Int, value)
    catch
        @warn "Failed to parse $key as integer: $value, using default: $default"
        return default
    end
end

"""
    get_env_var_bool(key::String, default::Bool)

Get environment variable as boolean with fallback to default value.
"""
function get_env_var_bool(key::String, default::Bool)
    value = get_env_var(key, string(default))
    if lowercase(value) in ["true", "1", "yes", "on"]
        return true
    elseif lowercase(value) in ["false", "0", "no", "off"]
        return false
    else
        @warn "Failed to parse $key as boolean: $value, using default: $default"
        return default
    end
end

# --------------------------------------------------------------
# Configuration Loading
# --------------------------------------------------------------

"""
    load_configuration()

Load and validate all configuration from environment variables.
"""
function load_configuration()
    # Load .env file if available
    load_env_file(".env")
    
    # Create configuration structures
    paths = PathConfig(
        get_env_var("LIMPS_HOME", pwd()),
        get_env_var("JULIA_PATH", Sys.BINDIR),
        get_env_var("PYTHON_PATH", "/usr/bin/python3")
    )
    
    language = LanguageConfig(
        get_env_var("JULIA_PROJECT", "."),
        get_env_var("JULIA_LOAD_PATH", ""),
        get_env_var("PYTHONPATH", "")
    )
    
    gpu = GPUConfig(
        get_env_var("CUDA_VISIBLE_DEVICES", "0"),
        get_env_var_bool("USE_GPU", true)
    )
    
    api = APIConfig(
        get_env_var("LIMPS_API_HOST", "localhost"),
        get_env_var_int("LIMPS_API_PORT", 8000)
    )
    
    database = DatabaseConfig(
        get_env_var("POSTGRES_DSN", "postgresql://localhost:5432/limps"),
        get_env_var("REDIS_URL", "redis://localhost:6379"),
        get_env_var("DATABASE_NAME", "limps")
    )
    
    logging = LoggingConfig(
        get_env_var("LOG_LEVEL", "INFO"),
        get_env_var("LOG_FILE", ""),
        get_env_var_bool("LOG_CONSOLE", true)
    )
    
    security = SecurityConfig(
        get_env_var("SECRET_KEY", "change-me-in-production"),
        get_env_var_int("TOKEN_EXPIRY", 3600),
        get_env_var_bool("ENABLE_SSL", false)
    )
    
    return MainConfig(paths, language, gpu, api, database, logging, security)
end

# --------------------------------------------------------------
# Legacy Compatibility
# --------------------------------------------------------------

"""
    build_legacy_config()

Build legacy configuration dictionary for backward compatibility.
"""
function build_legacy_config()
    cfg = load_configuration()
    
    legacy = Dict{String, String}()
    
    # Path configurations
    legacy["LIMPS_HOME"] = cfg.paths.home
    legacy["JULIA_PATH"] = cfg.paths.julia_path
    legacy["PYTHON_PATH"] = cfg.paths.python_path
    
    # Language configurations
    legacy["JULIA_PROJECT"] = cfg.language.julia_project
    legacy["JULIA_LOAD_PATH"] = cfg.language.julia_load_path
    legacy["PYTHONPATH"] = cfg.language.python_path
    
    # GPU configurations
    legacy["CUDA_VISIBLE_DEVICES"] = cfg.gpu.cuda_visible_devices
    legacy["USE_GPU"] = string(cfg.gpu.use_gpu)
    
    # API configurations
    legacy["LIMPS_API_HOST"] = cfg.api.host
    legacy["LIMPS_API_PORT"] = string(cfg.api.port)
    legacy["LIMPS_API_URL"] = api_url(cfg.api)
    
    # Database configurations
    legacy["POSTGRES_DSN"] = cfg.database.postgres_dsn
    legacy["REDIS_URL"] = cfg.database.redis_url
    legacy["DATABASE_NAME"] = cfg.database.database_name
    
    # Logging configurations
    legacy["LOG_LEVEL"] = cfg.logging.level
    legacy["LOG_FILE"] = cfg.logging.file
    legacy["LOG_CONSOLE"] = string(cfg.logging.console)
    
    # Security configurations
    legacy["SECRET_KEY"] = cfg.security.secret_key
    legacy["TOKEN_EXPIRY"] = string(cfg.security.token_expiry)
    legacy["ENABLE_SSL"] = string(cfg.security.enable_ssl)
    
    return legacy
end

# --------------------------------------------------------------
# Module Exports
# --------------------------------------------------------------

# Load configuration on module load
const config = load_configuration()
const CONFIG = build_legacy_config()

# Export main configuration
export config, CONFIG

# Export helper functions
export api_url, gpu_count, load_configuration, build_legacy_config

# Export configuration types for external use
export MainConfig, PathConfig, LanguageConfig, GPUConfig, APIConfig, 
       DatabaseConfig, LoggingConfig, SecurityConfig

end # module