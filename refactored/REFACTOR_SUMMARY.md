# Neutronics Surrogate Refactor Summary

## Overview

This refactor transforms the original scattered code into a robust, configurable, and maintainable system for building polynomial surrogate models from neutronics simulation data. The system now provides:

- **Unified CLI tool** for data processing and model fitting
- **Type-safe configuration management** for both Python and Julia
- **Comprehensive environment setup** with validation
- **LIMPS integration** with client generation
- **Production-ready error handling** and logging

## File Structure

```
refactored/
├── neutronics_surrogate.py      # Main CLI tool
├── limps_client.py              # Standalone LIMPS client
├── limps_env.py                 # Python configuration module
├── limps_env.jl                 # Julia configuration module
├── setup_limps_env.sh           # Environment setup script
├── .env.limps                   # Environment template
├── example_config.json          # Example configuration
├── requirements.txt             # Python dependencies
├── README_refactor.md           # Detailed documentation
└── REFACTOR_SUMMARY.md          # This summary
```

## Key Improvements

### 1. Robust Data Loading
- **Auto-delimiter detection**: Automatically handles CSV vs whitespace-separated files
- **Numeric coercion**: Safely converts data with graceful error handling
- **NaN management**: Intelligent handling of missing values
- **Geometry inference**: Attempts to detect square lattice dimensions

### 2. Polynomial Feature Engineering
- **Configurable degree**: Support for arbitrary polynomial degrees
- **Efficient implementation**: Uses `itertools.combinations_with_replacement`
- **Memory optimization**: Processes large datasets efficiently
- **Feature naming**: Clear, interpretable feature names

### 3. Ridge Regression
- **Closed-form solution**: Fast, stable matrix-based fitting
- **Regularization**: Configurable lambda parameter
- **Validation**: Comprehensive error checking

### 4. Configuration Management
- **Type-safe dataclasses**: Structured configuration with validation
- **Environment integration**: Automatic .env file loading
- **Backward compatibility**: Legacy dictionary interface
- **Cross-language support**: Python and Julia modules

### 5. LIMPS Integration
- **Payload generation**: Creates LIMPS-ready JSON
- **Client generation**: Optional Python client code
- **Coefficient export**: NPZ format for model persistence

## Usage Examples

### Basic Usage
```bash
python refactored/neutronics_surrogate.py \
  --raw data/raw.csv \
  --test data/test.csv \
  --outdir results
```

### Advanced Configuration
```bash
python refactored/neutronics_surrogate.py \
  --raw data/raw.csv \
  --test data/test.csv \
  --degree 3 \
  --max-input-cols 16 \
  --max-target-cols 24 \
  --max-rows 10000 \
  --lambda 1e-5 \
  --emit-client \
  --host my-limps-server.com \
  --port 9000 \
  --outdir results
```

### Environment Setup
```bash
# Set up environment
source refactored/setup_limps_env.sh

# Test Python configuration
python -c "from refactored.limps_env import CONFIG; print('✓ Python config loaded')"

# Test Julia configuration
julia -e 'include("refactored/limps_env.jl"); println("✓ Julia config loaded")'
```

## Output Files

The system generates several output files:

1. **`polynomial_surrogate_coefficients.npz`**: Fitted coefficients and feature names
2. **`limps_payload.json`**: LIMPS-ready optimization payload
3. **`fit_report.json`**: Comprehensive fitting report with statistics
4. **`limps_client.py`**: Generated client code (if requested)

## Configuration Features

### Command Line Arguments
- `--raw`: Path to RAW matrix file (required)
- `--test`: Path to TEST matrix file (required)
- `--degree`: Polynomial degree (default: 2)
- `--max-input-cols`: Cap input columns (default: 8)
- `--max-target-cols`: Cap target columns (default: 12)
- `--max-rows`: Cap rows for fitting (default: 5000)
- `--lambda`: Ridge regularization (default: 1e-6)
- `--outdir`: Output directory (default: ./out)
- `--emit-client`: Generate LIMPS client
- `--host`: Host for client (default: localhost)
- `--port`: Port for client (default: 8081)

### Environment Variables
The system supports comprehensive environment configuration:
- Core paths (LIMPS_HOME, LIMPS_JULIA_PATH, LIMPS_PYTHON_PATH)
- Language environments (JULIA_PROJECT, PYTHONPATH)
- GPU configuration (CUDA_VISIBLE_DEVICES, USE_GPU)
- API configuration (LIMPS_API_HOST, LIMPS_API_PORT)
- Database configuration (POSTGRES_DSN, REDIS_URL)
- Monitoring configuration (PROMETHEUS_PORT, GRAFANA_PORT)

## Error Handling

The refactored system includes comprehensive error handling:

- **File validation**: Clear error messages for missing/invalid files
- **Data validation**: Graceful handling of non-numeric data
- **Memory management**: Configurable limits to prevent overflow
- **Configuration validation**: Type checking and path validation
- **Network errors**: Timeout handling in generated clients

## Performance Considerations

- **Memory usage**: Configurable row/column limits
- **Computation time**: Polynomial feature count grows as O(d^degree)
- **Storage efficiency**: NPZ format for coefficient storage
- **Network optimization**: Configurable timeouts for API calls

## Dependencies

### Required
- `numpy`: Numerical computations
- `pandas`: Data loading and manipulation
- `requests`: HTTP client (for generated clients)

### Optional
- `python-dotenv`: Environment variable loading
- `scikit-learn`: Additional ML utilities
- `matplotlib`/`seaborn`: Plotting capabilities

## Testing

The system includes built-in testing capabilities:

```bash
# Test Python configuration
python refactored/limps_env.py

# Test Julia configuration
julia refactored/limps_env.jl

# Test environment setup
./refactored/setup_limps_env.sh
```

## Migration Guide

### From Original Code
1. Replace scattered data loading with `neutronics_surrogate.py`
2. Use `limps_env.py` for Python configuration
3. Use `limps_env.jl` for Julia configuration
4. Run `setup_limps_env.sh` for environment setup

### Configuration Migration
- Old environment variables are still supported
- New structured configuration provides better type safety
- Legacy dictionary interface maintains backward compatibility

## Future Enhancements

Potential improvements for future versions:

1. **Parallel processing**: Multi-core polynomial feature generation
2. **GPU acceleration**: CUDA-based matrix operations
3. **Model persistence**: Save/load fitted models
4. **Cross-validation**: Built-in model validation
5. **Feature selection**: Automatic feature importance ranking
6. **Web interface**: REST API for model serving
7. **Docker support**: Containerized deployment
8. **CI/CD integration**: Automated testing and deployment

## Conclusion

This refactor transforms a collection of scattered scripts into a production-ready, maintainable system. The unified CLI tool, type-safe configuration, and comprehensive error handling make it suitable for both research and production use. The modular design allows for easy extension and customization while maintaining backward compatibility.