import torch, torch.nn as nn, torch.nn.functional as F

class CompoundNode(nn.Module):
    def __init__(self, in_f, out_f, kinds=("linear","relu","sigmoid")):
        super().__init__()
        self.branches = nn.ModuleDict()
        for k in kinds:
            if k == "linear":
                lin = nn.Linear(in_f, out_f)
                with torch.no_grad():
                    lin.weight.zero_(); lin.bias.zero_()
                self.branches[k] = lin
            else:
                self.branches[k] = nn.Sequential(nn.Linear(in_f, out_f), getattr(nn, {"relu":"ReLU","sigmoid":"Sigmoid"}[k])())
        self.mix_logits = nn.Parameter(torch.zeros(len(kinds)))
        self.kinds = kinds

    def forward(self, x):
        ws = F.softmax(self.mix_logits, dim=0)
        out = 0
        for i,k in enumerate(self.kinds):
            out = out + ws[i]*self.branches[k](x)
        return out

class SkipPreserveBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.new = CompoundNode(in_f, out_f, kinds=("linear","relu"))
        self.skip = nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            self.skip.weight.zero_()

    def forward(self, x):
        return self.new(x) + self.skip(x)

class GradNormalizer:
    def __init__(self, modules, norm="max", eps=1e-8):
        self.modules = list(modules)
        self.norm = norm; self.eps = eps

    def _norm(self, params):
        if self.norm == "l2":
            s = 0.0
            for p in params:
                if p.grad is not None:
                    s += (p.grad.detach()**2).sum()
            return (s + 1e-12)**0.5
        m = 0.0
        for p in params:
            if p.grad is not None:
                m = max(m, float(p.grad.detach().abs().max().item()))
        return m

    @torch.no_grad()
    def apply(self):
        prev_s = None
        for m in self.modules:
            params = [p for p in m.parameters() if p.grad is not None]
            if not params: continue
            s = self._norm(params)
            if prev_s is not None and s > 0:
                scale = max(min(prev_s/(s+self.eps), 10.0), 0.1)
                for p in params: p.grad.mul_(scale)
            prev_s = s