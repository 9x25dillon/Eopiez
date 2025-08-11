from typing import Optional

def generate_sql(q: str, table: str = "corpus", col: str = "text", top_k: int = 50):
    prefix = (q or "").split(" ")[0].replace("'","''")
    return f"SELECT * FROM {table} WHERE {col} ILIKE '{prefix}%' LIMIT {top_k};"