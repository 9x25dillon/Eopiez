import torch
from ml2_core import GradNormalizer, SkipPreserveBlock

def train_step(model, loss_fn, batch, optimizer):
    x,y = batch
    out = model(x)
    loss = loss_fn(out,y)
    loss.backward()
    layers = [m for m in model.modules() if isinstance(m, (SkipPreserveBlock,))]
    GradNormalizer(layers, norm="max").apply()
    optimizer.step(); optimizer.zero_grad()
    return float(loss.item())