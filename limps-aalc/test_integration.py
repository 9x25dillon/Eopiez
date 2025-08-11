#!/usr/bin/env python3
"""
Integration test script for LIMPS-AALC monorepo
Tests the core functionality without requiring full dependencies
"""

import sys
import os
import json
from pathlib import Path

# Add the service paths to Python path
limps_root = Path(__file__).parent
admin_api_path = limps_root / "services" / "admin-api"
choppy_backend_path = limps_root / "services" / "choppy-backend"

sys.path.insert(0, str(admin_api_path))
sys.path.insert(0, str(choppy_backend_path))

def test_admin_api_modules():
    """Test admin-api module functionality"""
    print("Testing admin-api modules...")
    
    # Test ds_adapter
    from ds_adapter import simple_prefix_sql
    sql = simple_prefix_sql("test query", k=10)
    assert "ILIKE 'test%'" in sql
    assert "LIMIT 10" in sql
    print("✓ ds_adapter.simple_prefix_sql works")
    
    # Test coach
    from coach import coach_update
    original_state = {"lr": 1e-3, "top_k": 50, "entropy_floor": 3.0}
    state = original_state.copy()  # Copy to avoid mutation
    metrics = {"dev_loss_delta": 0.1}
    entropy_report = {"avg_token_entropy": 2.5}
    
    updated_state = coach_update(metrics, entropy_report, state)
    assert updated_state["lr"] < original_state["lr"]  # Should decrease lr
    assert updated_state["top_k"] > original_state["top_k"]  # Should increase top_k
    print("✓ coach.coach_update works")
    
    print("✓ Admin-api modules test passed")

def test_choppy_backend_modules():
    """Test choppy-backend module functionality"""
    print("Testing choppy-backend modules...")
    
    # Test chunker
    from chunker import chunk_text
    text = "This is a test document with many words that should be chunked properly"
    chunks = chunk_text(text, max_tokens=5, overlap=0.2)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
    print("✓ chunker.chunk_text works")
    
    # Test entropy_lamps
    from entropy_lamps import entropy_report, token_entropy
    entropy = token_entropy("hello world")
    assert entropy > 0
    
    report = entropy_report(["hello", "world", "test"])
    assert "avg" in report and "min" in report and "max" in report
    print("✓ entropy_lamps works")
    
    # Test chaos_sql
    from adapters.chaos_sql import generate_sql
    sql = generate_sql("search term", top_k=25)
    assert "ILIKE 'search%'" in sql
    assert "LIMIT 25" in sql
    print("✓ chaos_sql.generate_sql works")
    
    print("✓ Choppy-backend modules test passed")

def test_al_uls_client():
    """Test AL-ULS client structure"""
    print("Testing al-uls-client...")
    
    al_uls_path = limps_root / "services" / "al-uls-client"
    sys.path.insert(0, str(al_uls_path))
    
    # Mock websockets and httpx to test structure
    import types
    sys.modules['websockets'] = types.ModuleType('websockets')
    sys.modules['httpx'] = types.ModuleType('httpx')
    
    class MockAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json=None):
            class MockResponse:
                def raise_for_status(self): pass
                def json(self): return {"result": ["simplified"]}
            return MockResponse()
    
    sys.modules['httpx'].AsyncClient = MockAsyncClient
    
    from aluls_client import ALULSClient
    client = ALULSClient()
    assert client.base_http == "http://julia-ref:8008"
    assert client.ws_url == "ws://julia-ref:8008/ws"
    print("✓ AL-ULS client structure is correct")

def test_file_structure():
    """Test that all required files exist"""
    print("Testing file structure...")
    
    required_files = [
        "README.md",
        "db/migrations/001_init.sql",
        "services/admin-api/app.py",
        "services/admin-api/requirements.txt",
        "services/choppy-backend/main.py",
        "services/choppy-backend/requirements.txt",
        "services/al-uls-client/aluls_client.py",
        "services/julia-ref/server.jl",
        "services/julia-ref/Project.toml",
        "deploy/docker-compose.yml",
        "deploy/.env.example",
        ".tooling/pre-commit.sample"
    ]
    
    for file_path in required_files:
        full_path = limps_root / file_path
        assert full_path.exists(), f"Missing required file: {file_path}"
    
    print("✓ All required files exist")

def test_database_schema():
    """Test database schema"""
    print("Testing database schema...")
    
    schema_file = limps_root / "db" / "migrations" / "001_init.sql"
    schema_content = schema_file.read_text()
    
    required_tables = ["aalc_queue", "repo_rml", "repo_rfv", "aalc_snapshot", "ds_query_log"]
    for table in required_tables:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema_content
    
    print("✓ Database schema is correct")

def main():
    """Run all tests"""
    print("LIMPS-AALC Integration Test")
    print("=" * 40)
    
    try:
        test_file_structure()
        test_database_schema()
        test_admin_api_modules()
        test_choppy_backend_modules()
        test_al_uls_client()
        
        print("\n" + "=" * 40)
        print("✓ ALL TESTS PASSED!")
        print("\nLIMPS-AALC monorepo is properly implemented!")
        print("\nNext steps:")
        print("1. Run 'cd deploy && docker compose up --build' to start all services")
        print("2. Test endpoints at:")
        print("   - Admin API: http://localhost:8080")
        print("   - Choppy Backend: http://localhost:8090") 
        print("   - Julia Ref: http://localhost:8008")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())