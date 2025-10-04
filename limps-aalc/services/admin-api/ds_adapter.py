def simple_prefix_sql(q: str, table="corpus", col="text", k=50):
    prefix = (q or "").strip().split(" ")[0].replace("'", "''")
    return f"SELECT * FROM {table} WHERE {col} ILIKE '{prefix}%' LIMIT {k};"