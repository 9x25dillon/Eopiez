# Neutronics Surrogate (Refactor)

This refactor packages the data ingestion, polynomial feature expansion, ridge fitting,
and LIMPS-payload export into a single configurable CLI tool.

## Quickstart

```bash
python refactored/neutronics_surrogate.py \
  --raw "/mnt/data/lattice_physics_extracted/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)(1)/raw.csv" \
  --test "/mnt/data/lattice_physics_extracted/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)(1)/test.csv" \
  --degree 2 \
  --max-input-cols 8 \
  --max-target-cols 12 \
  --max-rows 5000 \
  --lambda 1e-6 \
  --outdir "refactored/out" \
  --emit-client
```

## Features

### Robust Data Loading
- **Auto-delimiter detection**: Automatically detects CSV vs whitespace-separated files
- **Numeric coercion**: Safely converts data to numeric types, handling errors gracefully
- **NaN handling**: Drops rows with missing values in either input or target matrices
- **Geometry inference**: Attempts to infer square lattice dimensions for diagnostics

### Polynomial Feature Engineering
- **Configurable degree**: Build polynomial features up to degree N (default: 2)
- **Efficient implementation**: Uses `itertools.combinations_with_replacement` to avoid duplicate terms
- **Memory efficient**: Processes features in chunks to handle large datasets

### Ridge Regression
- **Closed-form solution**: Uses matrix algebra for fast, stable fitting
- **Regularization**: Configurable lambda parameter for ridge regularization
- **Validation**: Comprehensive error checking and validation

### LIMPS Integration
- **Payload generation**: Creates LIMPS-ready JSON payloads
- **Client generation**: Optionally generates Python client code
- **Coefficient export**: Saves fitted coefficients in NPZ format

## Output Files

The script generates several output files:

1. **`polynomial_surrogate_coefficients.npz`**: Fitted coefficients and feature names
2. **`limps_payload.json`**: LIMPS-ready optimization payload
3. **`fit_report.json`**: Comprehensive fitting report with statistics
4. **`limps_client.py`**: Generated client code (if `--emit-client` is used)

## Configuration

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--raw` | str | required | Path to RAW matrix file |
| `--test` | str | required | Path to TEST matrix file |
| `--degree` | int | 2 | Polynomial degree (1..N) |
| `--max-input-cols` | int | 8 | Cap number of input columns |
| `--max-target-cols` | int | 12 | Cap number of target columns |
| `--max-rows` | int | 5000 | Cap number of rows for fitting |
| `--lambda` | float | 1e-6 | Ridge regularization parameter |
| `--outdir` | str | ./out | Output directory |
| `--emit-client` | flag | False | Generate LIMPS client code |
| `--host` | str | localhost | Host for generated client |
| `--port` | int | 8081 | Port for generated client |

### Example Configuration File

```json
{
  "raw": "/path/to/raw.csv",
  "test": "/path/to/test.csv",
  "degree": 2,
  "max_input_cols": 8,
  "max_target_cols": 12,
  "max_rows": 5000,
  "lambda": 1e-6,
  "outdir": "./out",
  "emit_client": true,
  "host": "localhost",
  "port": 8081
}
```

## Usage Examples

### Basic Usage
```bash
python refactored/neutronics_surrogate.py \
  --raw data/raw.csv \
  --test data/test.csv \
  --outdir results
```

### High-Degree Polynomial
```bash
python refactored/neutronics_surrogate.py \
  --raw data/raw.csv \
  --test data/test.csv \
  --degree 3 \
  --lambda 1e-5 \
  --outdir results
```

### Large Dataset
```bash
python refactored/neutronics_surrogate.py \
  --raw data/raw.csv \
  --test data/test.csv \
  --max-rows 10000 \
  --max-input-cols 16 \
  --max-target-cols 24 \
  --outdir results
```

### With Client Generation
```bash
python refactored/neutronics_surrogate.py \
  --raw data/raw.csv \
  --test data/test.csv \
  --emit-client \
  --host my-limps-server.com \
  --port 9000 \
  --outdir results
```

## Using the Generated Client

After running with `--emit-client`, you can use the generated client:

```python
from refactored.out.limps_client import PolyOptimizerClient

# Create client
client = PolyOptimizerClient(host="localhost", port=8081)

# Load payload
with open("refactored/out/limps_payload.json", "r") as f:
    payload = json.load(f)

# Send optimization request
result = client.optimize_polynomials(
    matrix=payload["matrix"],
    variables=payload["variables"],
    degree_limit=payload.get("degree_limit"),
    coeff_threshold=0.15
)

print(json.dumps(result, indent=2))
```

## Error Handling

The script includes comprehensive error handling:

- **File not found**: Clear error messages for missing input files
- **Invalid data**: Graceful handling of non-numeric data
- **Empty datasets**: Validation that sufficient data remains after cleaning
- **Memory issues**: Configurable limits to prevent memory overflow
- **Network errors**: Timeout handling in generated clients

## Performance Considerations

- **Memory usage**: Configurable row/column limits prevent memory issues
- **Computation time**: Polynomial feature count grows as O(d^degree)
- **Storage**: NPZ format for efficient coefficient storage
- **Network**: Configurable timeouts for LIMPS API calls

## Dependencies

Required Python packages:
- `numpy`: Numerical computations
- `pandas`: Data loading and manipulation
- `requests`: HTTP client (for generated clients)

Optional:
- `python-dotenv`: Environment variable loading

## Troubleshooting

### Common Issues

1. **"No valid finite rows after cleaning"**
   - Check input file format
   - Increase `--max-rows` if dataset is small
   - Verify numeric data in input files

2. **Memory errors with large datasets**
   - Reduce `--max-rows` or `--max-input-cols`
   - Use lower polynomial degree
   - Increase system memory

3. **Poor fit quality**
   - Try different `--lambda` values
   - Increase polynomial degree
   - Check data quality and preprocessing

### Debug Mode

For debugging, you can examine the generated `fit_report.json`:

```bash
cat refactored/out/fit_report.json | jq .
```

This shows detailed information about the fitting process, including:
- Input/output shapes
- RMSE for each target
- Feature counts
- File paths