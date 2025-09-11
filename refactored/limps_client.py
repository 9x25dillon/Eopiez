import requests, json

class PolyOptimizerClient:
    def __init__(self, host="localhost", port=8081):
        self.url = f"http://{host}:{port}/optimize"

    def optimize_polynomials(self, matrix, variables, degree_limit=None, min_rank=None,
                             structure=None, coeff_threshold=0.15, chebyshev=False, timeout=30):
        payload = {
            "matrix": matrix,
            "variables": variables,
            "coeff_threshold": coeff_threshold,
            "chebyshev": chebyshev,
        }
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
    client = PolyOptimizerClient("localhost", 8081)
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