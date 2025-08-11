def coach_update(metrics, entropy_report, state):
    if metrics.get("dev_loss_delta", 0.0) > -1e-3:
        state["lr"] = state.get("lr",1e-3) * 0.8
    if entropy_report.get("avg_token_entropy", 9.9) < state.get("entropy_floor", 3.0):
        state["top_k"] = min(state.get("top_k", 50)+10, 200)
    return state