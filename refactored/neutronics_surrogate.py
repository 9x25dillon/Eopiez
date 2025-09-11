#!/usr/bin/env python3
"""
Neutronics Surrogate Builder (Refactored)
----------------------------------------
- Loads RAW and TEST matrices (whitespace or CSV)
- Coerces to numeric and drops rows with NaNs
- Optionally infers lattice geometry (for diagnostics)
- Builds polynomial feature map up to degree N (default 2)
- Fits ridge-regularized least squares (closed-form)
- Exports LIMPS-ready payload and a coefficients NPZ
- (Optional) Generates a minimal Python client for your LIMPS server

Usage:
  python neutronics_surrogate.py \
      --raw /path/to/raw.csv \
      --test /path/to/test.csv \
      --degree 2 \
      --max-input-cols 8 \
      --max-target-cols 12 \
      --max-rows 5000 \
      --lambda 1e-6 \
      --outdir ./out \
      --emit-client

See example_config.json for config-based invocation.
"""

import os, json, math, re, itertools, argparse
from pathlib import Path
import numpy as np
import pandas as pd

def detect_delimiter(sample_lines):
    """
    Heuristics:
    - If we find commas across lines consistently -> comma
    - Else fallback to whitespace (\\s+)
    """
    comma_count = sum(line.count(',') for line in sample_lines)
    if comma_count >= max(3, len(sample_lines)):  # crude threshold
        return ','
    return r"\\s+"

def load_matrix(path, max_preview_lines=5):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = [next(f) for _ in range(max_preview_lines)]
    delim = detect_delimiter(sample)
    if delim == ',':
        df = pd.read_csv(path, header=None)
    else:
        df = pd.read_csv(path, header=None, sep=delim, engine="python")
    # Coerce to numeric, keep NaNs for now
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, delim

def coerce_dropna_pair(X_df, Y_df):
    # align rows, then drop any row with NaN in either
    n = min(len(X_df), len(Y_df))
    X = X_df.iloc[:n, :].copy()
    Y = Y_df.iloc[:n, :].copy()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    Xc = X[mask].to_numpy(dtype=float)
    Yc = Y[mask].to_numpy(dtype=float)
    return Xc, Yc

def infer_square_dim(n_cols):
    r = int(math.isqrt(n_cols))
    return r if r*r == n_cols else None

def poly_feature_names(d, degree):
    # Generate names for monomials up to given degree using combinations with replacement
    names = ["1"]
    # degree 1..N
    for deg in range(1, degree+1):
        for comb in itertools.combinations_with_replacement(range(d), deg):
            # name like x1*x3*x3 for [0,2,2]
            term = "*".join(f"x{i+1}" for i in comb)
            names.append(term)
    return names

def poly_features(X, degree=2):
    """
    Build polynomial features up to 'degree' without permutations.
    Uses combinations_with_replacement to avoid duplicates.
    Returns (Phi, names)
    """
    n, d = X.shape
    feats = [np.ones((n,1))]
    # degree 1..N
    names = ["1"]
    for deg in range(1, degree+1):
        for comb in itertools.combinations_with_replacement(range(d), deg):
            col = np.prod([X[:, i] for i in comb], axis=0).reshape(n,1)
            feats.append(col)
            names.append("*".join(f"x{i+1}" for i in comb))
    Phi = np.hstack(feats)
    return Phi, names

def ridge_closed_form(Phi, Y, lam=1e-6):
    PtP = Phi.T @ Phi
    PtY = Phi.T @ Y
    # Regularize
    B = np.linalg.solve(PtP + lam*np.eye(PtP.shape[0]), PtY)
    return B

def rmse_columns(pred, Y):
    return np.sqrt(np.mean((pred - Y)**2, axis=0))

def build_payload(X_used, variables, degree_limit, min_rank, structure, coeff_threshold, chebyshev, rmse_first10, n_targets):
    return {
        "matrix": X_used.tolist(),
        "variables": variables,
        "degree_limit": degree_limit,
        "min_rank": min_rank,
        "structure": structure,
        "coeff_threshold": coeff_threshold,
        "chebyshev": chebyshev,
        "targets_preview": {
            "n_targets_used": int(n_targets),
            "rmse_first10": [float(x) for x in rmse_first10]
        }
    }

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raw", required=True, help="Path to RAW matrix file (whitespace or CSV)")
    p.add_argument("--test", required=True, help="Path to TEST matrix file (whitespace or CSV)")
    p.add_argument("--degree", type=int, default=2, help="Polynomial degree (1..N)")
    p.add_argument("--max-input-cols", type=int, default=8, help="Cap number of input columns from RAW")
    p.add_argument("--max-target-cols", type=int, default=12, help="Cap number of target columns from TEST")
    p.add_argument("--max-rows", type=int, default=5000, help="Cap number of rows used for fitting")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-6, help="Ridge regularization lambda")
    p.add_argument("--outdir", default="./out", help="Output directory")
    p.add_argument("--emit-client", action="store_true", help="Also emit a minimal Python client for LIMPS")
    p.add_argument("--host", default="localhost", help="Host for emitted client")
    p.add_argument("--port", type=int, default=8081, help="Port for emitted client")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_df, raw_delim = load_matrix(args.raw)
    test_df, test_delim = load_matrix(args.test)

    # Info / Geometry
    info = {
        "raw_shape": raw_df.shape,
        "test_shape": test_df.shape,
        "raw_delimiter": raw_delim,
        "test_delimiter": test_delim,
        "raw_square_dim": infer_square_dim(raw_df.shape[1]),
        "test_square_dim": infer_square_dim(test_df.shape[1]),
        "degree": int(args.degree),
        "max_input_cols": int(args.max_input_cols),
        "max_target_cols": int(args.max_target_cols),
        "max_rows": int(args.max_rows),
        "lambda": float(args.lam),
    }

    # Cap rows and columns
    n_rows = min(args.max_rows, raw_df.shape[0], test_df.shape[0])
    X_df = raw_df.iloc[:n_rows, :args.max_input_cols]
    Y_df = test_df.iloc[:n_rows, :args.max_target_cols]

    X_used, Y_used = coerce_dropna_pair(X_df, Y_df)

    if X_used.size == 0 or Y_used.size == 0:
        raise RuntimeError("No valid finite rows after cleaning. Check file formatting or increase caps.")

    # Build polynomial features
    Phi, feat_names = poly_features(X_used, degree=args.degree)

    # Fit ridge
    B = ridge_closed_form(Phi, Y_used, lam=args.lam)

    # Predictions and RMSE
    pred = Phi @ B
    rmse = rmse_columns(pred, Y_used).tolist()

    # Save coefficients
    coef_path = outdir / "polynomial_surrogate_coefficients.npz"
    np.savez(coef_path, B=B, feat_names=np.array(feat_names, dtype=object))

    # Build LIMPS payload
    variables = [f"x{i+1}" for i in range(X_used.shape[1])]
    payload = build_payload(
        X_used=X_used,
        variables=variables,
        degree_limit=args.degree,
        min_rank=None,
        structure="sparse",
        coeff_threshold=0.15,
        chebyshev=False,
        rmse_first10=rmse[:10],
        n_targets=Y_used.shape[1],
    )
    payload_path = outdir / "limps_payload.json"
    with payload_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # Emit client if requested
    if args.emit_client:
        client_code = f'''import requests, json

class PolyOptimizerClient:
    def __init__(self, host="{args.host}", port={args.port}):
        self.url = f"http://{{host}}:{{port}}/optimize"

    def optimize_polynomials(self, matrix, variables, degree_limit=None, min_rank=None,
                             structure=None, coeff_threshold=0.15, chebyshev=False, timeout=30):
        payload = {{
            "matrix": matrix,
            "variables": variables,
            "coeff_threshold": coeff_threshold,
            "chebyshev": chebyshev,
        }}
        if degree_limit is not None:
            payload["degree_limit"] = degree_limit
        if min_rank is not None:
            payload["min_rank"] = min_rank
        if structure is not None:
            payload["structure"] = structure

        resp = requests.post(self.url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

if __name__ == "__main__":
    with open("limps_payload.json", "r") as f:
        payload = json.load(f)
    client = PolyOptimizerClient()
    out = client.optimize_polynomials(
        matrix=payload["matrix"],
        variables=payload["variables"],
        degree_limit=payload.get("degree_limit"),
        min_rank=payload.get("min_rank"),
        structure=payload.get("structure"),
        coeff_threshold=payload.get("coeff_threshold", 0.15),
        chebyshev=payload.get("chebyshev", False),
    )
    print(json.dumps(out, indent=2))
'''
        client_path = outdir / "limps_client.py"
        with client_path.open("w") as f:
            f.write(client_code)

    # Save info and report
    report = {
        "info": info,
        "rmse_first10": rmse[:10],
        "n_samples_fit": int(Phi.shape[0]),
        "n_features": int(Phi.shape[1]),
        "n_targets_fit": int(Y_used.shape[1]),
        "coef_path": str(coef_path),
        "payload_path": str(payload_path),
    }
    report_path = outdir / "fit_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()