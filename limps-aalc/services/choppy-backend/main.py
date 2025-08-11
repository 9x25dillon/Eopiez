from fastapi import FastAPI, UploadFile, File
from chunker import chunk_text
from entropy_lamps import entropy_report
from adapters.chaos_sql import generate_sql

app = FastAPI(title="Choppy Backend")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), overlap: float = 0.15):
    text = (await file.read()).decode("utf-8", errors="ignore")
    chunks = chunk_text(text, overlap=overlap)
    ent = entropy_report(chunks)
    return {"chunks": chunks[:5], "num_chunks": len(chunks), "entropy": ent}

@app.post("/chunks/query")
async def query_chunks(q: str, top_k: int = 50):
    sql = generate_sql(q, top_k=top_k)
    return {"sql": sql}