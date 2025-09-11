"""
limps_env.py - Refactored LIMPS Environment Configuration

Centralized, type-checked configuration for LIMPS Python code with improved
error handling, validation, and documentation.

Usage:
    from limps_env import CONFIG, get_config, validate_config
    
    # Access configuration
    print(CONFIG.api_url)
    
    # Get with default
    debug_mode = get_config("DEBUG", False)
    
    # Validate configuration
    validate_config()
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
import warnings


# ----------------------------------------------------------------------
# Configuration Data Classes
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    """Core path configurations."""
    home: Path
    julia_path: Path
    python_path: Path
    
    def __post_init__(self):
        """Validate paths exist."""
        for field_name, path in self.__dict__.items():
            if not path.exists():
                warnings.warn(f"Path does not exist: {field_name} = {path}")


@dataclass(frozen=True)
class LanguageConfig:
    """Language environment configurations."""
    julia_project: str
    julia_load_path: str
    python_path: str


@dataclass(frozen=True)
class GPUConfig:
    """GPU and CUDA configuration."""
    cuda_visible_devices: str
    use_gpu: bool
    
    @property
    def gpu_count(self) -> int:
        """Return number of visible GPU devices."""
        if not self.use_gpu or not self.cuda_visible_devices:
            return 0
        return len(self.cuda_visible_devices.split(','))


@dataclass(frozen=True)
class APIConfig:
    """API server configuration."""
    host: str
    port: int
    
    @property
    def url(self) -> str:
        """Return full API URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def base_url(self) -> str:
        """Alias for url property."""
        return self.url


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""
    postgres_dsn: str
    redis_url: str
    
    @property
    def postgres_components(self) -> Dict[str, str]:
        """Parse PostgreSQL DSN components."""
        # Simple parser for postgresql://user:pass@host:port/db
        if not self.postgres_dsn.startswith('postgresql://'):
            return {}
        
        try:
            # Remove protocol
            url = self.postgres_dsn[13:]  # len('postgresql://')
            
            # Split user:pass@host:port/db
            if '@' in url:
                auth, rest = url.split('@', 1)
                user, password = auth.split(':', 1) if ':' in auth else (auth, '')
            else:
                user, password, rest = '', '', url
            
            if '/' in rest:
                host_port, database = rest.rsplit('/', 1)
            else:
                host_port, database = rest, ''
            
            if ':' in host_port:
                host, port = host_port.rsplit(':', 1)
            else:
                host, port = host_port, '5432'
            
            return {
                'user': user,
                'password': password,
                'host': host,
                'port': port,
                'database': database
            }
        except Exception:
            return {}


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring service configuration."""
    prometheus_port: int
    grafana_port: int


@dataclass(frozen=True)
class LIMPSConfig:
    """Complete LIMPS configuration."""
    paths: PathConfig
    languages: LanguageConfig
    gpu: GPUConfig
    api: APIConfig
    database: DatabaseConfig
    monitoring: MonitoringConfig
    
    # Convenience properties for backward compatibility
    @property
    def limps_home(self) -> str:
        return str(self.paths.home)
    
    @property
    def limps_api_url(self) -> str:
        return self.api.url
    
    @property
    def postgres_dsn(self) -> str:
        return self.database.postgres_dsn
    
    @property
    def redis_url(self) -> str:
        return self.database.redis_url
    
    @property
    def use_gpu(self) -> bool:
        return self.gpu.use_gpu


# ----------------------------------------------------------------------
# Environment Loading
# ----------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env file if it exists and python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        
        # Look for .env.limps in current directory and parent directories
        current = Path.cwd()
        for path in [current] + list(current.parents):
            env_file = path / ".env.limps"
            if env_file.is_file():
                load_dotenv(dotenv_path=env_file, override=False)
                break
    except ImportError:
        # python-dotenv not installed, skip
        pass
    except Exception as e:
        warnings.warn(f"Failed to load .env file: {e}")


def _require_env(key: str) -> str:
    """Get required environment variable or raise informative error."""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Please run 'source ./setup_limps_env.sh' first."
        )
    return value.strip()


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    value = os.getenv(key, default)
    return value.strip() if value else default


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = _get_env(key, "").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    elif value in ("0", "false", "no", "off"):
        return False
    else:
        return default


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with validation."""
    value = _get_env(key)
    if not value:
        return default
    
    try:
        return int(value)
    except ValueError:
        warnings.warn(f"Invalid integer value for {key}: {value}, using default {default}")
        return default


def _create_config() -> LIMPSConfig:
    """Create configuration from environment variables."""
    _load_dotenv()
    
    # Core paths
    limps_home = Path(_require_env("LIMPS_HOME"))
    paths = PathConfig(
        home=limps_home,
        julia_path=Path(_require_env("LIMPS_JULIA_PATH")),
        python_path=Path(_require_env("LIMPS_PYTHON_PATH"))
    )
    
    # Language environments
    languages = LanguageConfig(
        julia_project=_require_env("JULIA_PROJECT"),
        julia_load_path=_get_env("JULIA_LOAD_PATH"),
        python_path=_get_env("PYTHONPATH")
    )
    
    # GPU configuration
    gpu = GPUConfig(
        cuda_visible_devices=_get_env("CUDA_VISIBLE_DEVICES", "0"),
        use_gpu=_get_env_bool("USE_GPU", True)
    )
    
    # API configuration
    api = APIConfig(
        host=_get_env("LIMPS_API_HOST", "localhost"),
        port=_get_env_int("LIMPS_API_PORT", 8000)
    )
    
    # Database configuration
    database = DatabaseConfig(
        postgres_dsn=_require_env("POSTGRES_DSN"),
        redis_url=_require_env("REDIS_URL")
    )
    
    # Monitoring configuration
    monitoring = MonitoringConfig(
        prometheus_port=_get_env_int("PROMETHEUS_PORT", 9090),
        grafana_port=_get_env_int("GRAFANA_PORT", 3000)
    )
    
    return LIMPSConfig(
        paths=paths,
        languages=languages,
        gpu=gpu,
        api=api,
        database=database,
        monitoring=monitoring
    )


# ----------------------------------------------------------------------
# Public Interface
# ----------------------------------------------------------------------

# Create the global configuration instance
try:
    CONFIG = _create_config()
except Exception as e:
    print(f"❌ Failed to load LIMPS configuration: {e}", file=sys.stderr)
    print("   Please run 'source ./setup_limps_env.sh' first.", file=sys.stderr)
    sys.exit(1)


def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value by key path.
    
    Args:
        key: Dot-separated path to configuration value (e.g., "api.port")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Examples:
        >>> get_config("api.port")
        8000
        >>> get_config("gpu.use_gpu")
        True
        >>> get_config("nonexistent.key", "default")
        'default'
    """
    try:
        obj = CONFIG
        for part in key.split('.'):
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        return default


def validate_config() -> None:
    """
    Validate the current configuration.
    
    Raises:
        EnvironmentError: If configuration is invalid
    """
    errors = []
    
    # Validate paths
    required_paths = [
        ("LIMPS_HOME", CONFIG.paths.home),
        ("LIMPS_JULIA_PATH", CONFIG.paths.julia_path),
        ("LIMPS_PYTHON_PATH", CONFIG.paths.python_path)
    ]
    
    for name, path in required_paths:
        if not path.exists():
            errors.append(f"Required path does not exist: {name} = {path}")
    
    # Validate ports
    port_configs = [
        ("API", CONFIG.api.port),
        ("Prometheus", CONFIG.monitoring.prometheus_port),
        ("Grafana", CONFIG.monitoring.grafana_port)
    ]
    
    for name, port in port_configs:
        if not (1 <= port <= 65535):
            errors.append(f"Invalid port number for {name}: {port}")
    
    # Validate URLs
    try:
        import urllib.parse
        for name, url in [("PostgreSQL DSN", CONFIG.database.postgres_dsn),
                         ("Redis URL", CONFIG.database.redis_url)]:
            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                errors.append(f"Invalid URL format for {name}: {url}")
    except ImportError:
        pass  # Skip URL validation if urllib is not available
    
    if errors:
        raise EnvironmentError(
            f"Configuration validation failed:\n" +
            "\n".join(f"  • {error}" for error in errors)
        )


def print_config() -> None:
    """Print current configuration in a readable format."""
    print("🚀 LIMPS Configuration")
    print("=" * 50)
    print(f"Home Directory: {CONFIG.paths.home}")
    print(f"API URL: {CONFIG.api.url}")
    print(f"GPU Enabled: {CONFIG.gpu.use_gpu}")
    print(f"GPU Devices: {CONFIG.gpu.cuda_visible_devices}")
    print(f"PostgreSQL: {CONFIG.database.postgres_dsn}")
    print(f"Redis: {CONFIG.database.redis_url}")
    print("=" * 50)


# ----------------------------------------------------------------------
# Backward Compatibility
# ----------------------------------------------------------------------

# Legacy dictionary interface for backward compatibility
class _LegacyConfigDict(dict):
    """Legacy dictionary interface that wraps the new dataclass config."""
    
    def __init__(self, config: LIMPSConfig):
        self._config = config
        # Populate with legacy keys
        super().__init__({
            "LIMPS_HOME": str(config.paths.home),
            "LIMPS_JULIA_PATH": str(config.paths.julia_path),
            "LIMPS_PYTHON_PATH": str(config.paths.python_path),
            "JULIA_PROJECT": config.languages.julia_project,
            "JULIA_LOAD_PATH": config.languages.julia_load_path,
            "PYTHONPATH": config.languages.python_path,
            "CUDA_VISIBLE_DEVICES": config.gpu.cuda_visible_devices,
            "USE_GPU": config.gpu.use_gpu,
            "LIMPS_API_HOST": config.api.host,
            "LIMPS_API_PORT": config.api.port,
            "LIMPS_API_URL": config.api.url,
            "POSTGRES_DSN": config.database.postgres_dsn,
            "REDIS_URL": config.database.redis_url,
            "PROMETHEUS_PORT": config.monitoring.prometheus_port,
            "GRAFANA_PORT": config.monitoring.grafana_port,
        })
    
    def __setitem__(self, key, value):
        raise TypeError("CONFIG is read-only")
    
    def __delitem__(self, key):
        raise TypeError("CONFIG is read-only")
    
    def clear(self):
        raise TypeError("CONFIG is read-only")
    
    def pop(self, *args):
        raise TypeError("CONFIG is read-only")
    
    def popitem(self):
        raise TypeError("CONFIG is read-only")
    
    def update(self, *args, **kwargs):
        raise TypeError("CONFIG is read-only")


# Create legacy dictionary interface
LEGACY_CONFIG = _LegacyConfigDict(CONFIG)

# For scripts that still use the old dictionary interface
def get(key: str, default: Any = None) -> Any:
    """Legacy function for backward compatibility."""
    return LEGACY_CONFIG.get(key, default)


# ----------------------------------------------------------------------
# Module Testing
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing LIMPS configuration...")
    try:
        validate_config()
        print_config()
        print("✅ Configuration is valid!")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)