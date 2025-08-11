import torch

def make_rfv_snapshot(model: torch.nn.Module, sample_batch, label_map=None):
    with torch.no_grad():
        feats = model(sample_batch).cpu()
    meta = {"arch":"torch", "shape": list(feats.shape), "labels": label_map}
    # In real usage: persist feats to object store and register via /rfv/publish
    return {"rdata_uri":"s3://bucket/rfv/xyz.pt", "rml_meta": meta}