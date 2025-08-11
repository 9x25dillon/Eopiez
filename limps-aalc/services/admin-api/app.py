from fastapi import FastAPI, Body
from datetime import datetime, timedelta
import uuid

from ds_adapter import simple_prefix_sql
from rfv import make_rfv_snapshot
from coach import coach_update

app = FastAPI(title="LIMPS AALC Admin/IA API")

@app.post("/pq/lease")
def lease_job(duration_minutes: int = 15):
    return {
        "job_id": str(uuid.uuid4()),
        "task_spec": {"dataset":"speech_conv_v1","objective":"ASR"},
        "ifv_spec": {"features":["mfcc","prosody"]},
        "leased_until": (datetime.utcnow()+timedelta(minutes=duration_minutes)).isoformat()+"Z"
    }

@app.post("/rfv/publish")
def publish_rfv(payload: dict = Body(...)):
    return {"rfv_id": str(uuid.uuid4())}

@app.post("/ds/select")
def ds_select(payload: dict = Body(...)):
    sql = simple_prefix_sql(payload.get("query",""), k=payload.get("top_k",50))
    return {"sql": sql, "result_uri": "s3://bucket/ds/123.parquet"}

@app.post("/ml2/train_step")
def ml2_train_step(payload: dict = Body(...)):
    # stubbed metrics; real impl loads result_uri, feeds model, applies GradNormalizer
    metrics = {"loss": 0.231, "wer": 10.4}
    state = {"lr": 1e-3, "top_k": 50, "entropy_floor": 3.0}
    state = coach_update({"dev_loss_delta": 0.0}, {"avg_token_entropy": 2.7}, state)
    return {"metrics": metrics, "state": state, "snapshot_id": str(uuid.uuid4())}