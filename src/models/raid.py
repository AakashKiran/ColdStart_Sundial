"""RAID model aligned with the corrected architecture/training notebook.

This is a paper-guided implementation, not a verified exact reproduction of
the RAID authors' code. Training uses the paper-objective variant without an
auxiliary final-forecast gate loss, as in the notebook.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angle = t.float().unsqueeze(1) * freq.unsqueeze(0)
    out = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
    return F.pad(out, (0, 1)) if dim % 2 else out


class GATv2Layer(nn.Module):
    """Dense masked GATv2 layer; suitable for the small 86-SKU graph."""
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("bottleneck_dim must be divisible by gat_heads")
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.src = nn.Linear(dim, dim, bias=False)
        self.dst = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, self.dh))
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, allowed):
        n = x.shape[0]
        src = self.src(x).view(n, self.heads, self.dh)
        dst = self.dst(x).view(n, self.heads, self.dh)
        pair = F.leaky_relu(src[:, None] + dst[None, :], 0.2)
        # logits[i, j, head] scores neighbor j when updating node i.
        logits = (pair * self.attn[None, None]).sum(-1)
        logits = logits.masked_fill(~allowed[:, :, None], torch.finfo(logits.dtype).min)
        alpha = self.drop(torch.softmax(logits, dim=1))  # [target_i, neighbor_j, head]
        msg = torch.einsum("ijh,jhd->ihd", alpha, dst).reshape(n, self.dim)
        return F.elu(self.proj(msg) + x)


class DiTBlock(nn.Module):
    def __init__(self, hidden=64, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(dropout)
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, cond):
        sa, sca, ga, sm, scm, gm = self.mod(cond).chunk(6, dim=-1)
        z = self.norm1(x) * (1 + sca[:, None]) + sa[:, None]
        x = x + ga[:, None] * self.attn(z, z, z, need_weights=False)[0]
        z = self.norm2(x) * (1 + scm[:, None]) + sm[:, None]
        return x + gm[:, None] * self.mlp(z)


class DiT(nn.Module):
    def __init__(self, horizon=13, bottleneck_dim=16, hidden=64, layers=4,
                 heads=4, time_dim=128, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.in_proj = nn.Linear(1, hidden)
        self.pos = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.context_proj = nn.Linear(bottleneck_dim, hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, 1)

    def forward(self, x, t, context):
        h = self.in_proj(x.unsqueeze(-1)) + self.pos
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))
        cond = t_emb + self.context_proj(context)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.norm(h)).squeeze(-1)


class RAID(nn.Module):
    def __init__(self, horizon=13, embedding_dim=1024, bottleneck_dim=16, k=20,
                 gat_heads=4, gat_layers=2, dit_hidden=64, dit_layers=4, dit_heads=4,
                 dit_dropout=0.1, time_dim=128, diffusion_steps=50,
                 beta_start=1e-4, beta_end=0.02, shape_clamp=10.0, epsilon=1e-6,
                 base_loss_weight=0.01, gate_bias_init=-3.0):
        super().__init__()
        self.horizon, self.k = horizon, k
        self.bottleneck_dim = bottleneck_dim
        self.shape_clamp, self.eps = shape_clamp, epsilon
        self.base_loss_weight = base_loss_weight
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, bottleneck_dim), nn.Tanh()
        )
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.gat = nn.ModuleList([GATv2Layer(bottleneck_dim, gat_heads) for _ in range(gat_layers)])
        self.scale_mean = nn.Sequential(
            nn.Linear(bottleneck_dim, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2)
        )
        self.dit = DiT(horizon, bottleneck_dim, dit_hidden, dit_layers, dit_heads, time_dim, dit_dropout)
        self.gate = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )
        nn.init.constant_(self.gate[-1].bias, gate_bias_init)
        beta = torch.linspace(beta_start, beta_end, diffusion_steps)
        self.register_buffer("betas", beta)
        self.register_buffer("alphas", 1.0 - beta)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - beta, dim=0))

    def graph(self, embeddings):
        raw = F.normalize(embeddings, dim=-1)
        sim = raw @ raw.T
        n = sim.shape[0]
        sim_no_self = sim.clone()
        sim_no_self.fill_diagonal_(-torch.inf)
        k = min(self.k, max(1, n - 1))
        neighbors = torch.topk(sim_no_self, k=k, dim=-1).indices
        allowed = torch.zeros_like(sim, dtype=torch.bool)
        allowed.scatter_(1, neighbors, True)
        allowed.fill_diagonal_(True)
        compact = self.bottleneck(embeddings)
        context = compact
        for layer in self.gat:
            context = layer(context, allowed)
        return raw, compact, context, allowed

    def base_forecast(self, target_raw, target_compact, donor_raw, donor_history,
                      donor_mean, donor_std, donor_mask):
        # Shapes: target [B,D], donor [B,K,D], history [B,K,H],
        # mean/std [B,K,1], mask [B,K].
        tau = F.softplus(self.raw_temperature).clamp_min(1e-4)
        logits = torch.einsum(
            "bd,bkd->bk", F.normalize(target_raw, dim=-1), F.normalize(donor_raw, dim=-1)
        ) / tau
        logits = logits.masked_fill(~donor_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * donor_mask.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        shapes = ((donor_history - donor_mean) / donor_std.clamp_min(self.eps))
        shapes = shapes.clamp(-self.shape_clamp, self.shape_clamp)
        z_agg = (weights.unsqueeze(-1) * shapes).sum(dim=1)
        params = self.scale_mean(target_compact)
        target_scale = F.softplus(params[:, :1]).clamp_min(self.eps)
        target_mean = params[:, 1:2]
        base = target_scale * z_agg + target_mean
        return base, weights, target_scale, target_mean

    def forward(self, embeddings, target_indices, donor_indices, donor_history,
                donor_mean, donor_std, donor_mask, target_values=None, target_mask=None):
        raw, compact, context, allowed = self.graph(embeddings)
        target_raw = raw[target_indices]
        target_compact = compact[target_indices]
        target_context = context[target_indices]
        donor_raw = raw[donor_indices]
        base, weights, target_scale, target_mean = self.base_forecast(
            target_raw, target_compact, donor_raw, donor_history, donor_mean, donor_std, donor_mask
        )
        result = {
            "base": base, "context": target_context, "weights": weights,
            "target_scale": target_scale, "target_mean": target_mean, "allowed": allowed
        }
        if target_values is not None:
            mask = target_mask.to(target_values.dtype)
            # The diffusion target is detached so diffusion loss does not update the base through its target.
            residual = ((target_values - base) * mask).detach()
            t = torch.randint(0, len(self.betas), (len(target_indices),), device=embeddings.device)
            noise = torch.randn_like(residual)
            ab = self.alpha_bars[t].unsqueeze(-1)
            noisy = ab.sqrt() * residual + (1 - ab).sqrt() * noise
            eps_hat = self.dit(noisy, t, target_context)
            denom = mask.sum().clamp_min(1.0)
            loss_diff = (((eps_hat - noise) ** 2) * mask).sum() / denom
            loss_base = (((base - target_values) ** 2) * mask).sum() / denom
            result.update({
                "loss": loss_diff + self.base_loss_weight * loss_base,
                "diffusion_loss": loss_diff.detach(),
                "base_loss": loss_base.detach(),
                "final_loss": torch.zeros((), device=embeddings.device)
            })
        return result

    @torch.no_grad()
    def predict_samples(self, embeddings, target_index, donor_indices, donor_history,
                        donor_mean, donor_std, num_samples=20, inference_steps=10):
        # Do not leave the caller's model in eval mode as a side effect.
        was_training = self.training
        self.eval()
        try:
            device = embeddings.device
            raw, compact, context, _ = self.graph(embeddings)
            ti = torch.tensor([target_index], device=device, dtype=torch.long)
            di = torch.tensor(donor_indices, device=device, dtype=torch.long).unsqueeze(0)
            dh = donor_history.to(device).unsqueeze(0)
            dm = donor_mean.to(device).reshape(1, -1, 1)
            ds = donor_std.to(device).reshape(1, -1, 1)
            mask = torch.ones((1, len(donor_indices)), dtype=torch.bool, device=device)
            base, _, _, _ = self.base_forecast(raw[ti], compact[ti], raw[di], dh, dm, ds, mask)
            c = context[ti].expand(num_samples, -1)
            r = torch.randn((num_samples, self.horizon), device=device)
            count = min(inference_steps, len(self.betas))
            steps = torch.linspace(len(self.betas) - 1, 0, count, device=device).round().long().unique_consecutive()
            for i, ts in enumerate(steps):
                t = torch.full((num_samples,), int(ts.item()), device=device, dtype=torch.long)
                eps_hat = self.dit(r, t, c)
                ab_t = self.alpha_bars[ts]
                r0 = (r - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt().clamp_min(1e-5)
                ab_prev = (self.alpha_bars[steps[i + 1]] if i + 1 < len(steps)
                           else torch.ones((), device=device))
                alpha_step = (ab_t / ab_prev).clamp(1e-8, 1.0)
                beta_step = 1.0 - alpha_step
                denom = (1.0 - ab_t).clamp_min(1e-8)
                mean = (
                    ab_prev.sqrt() * beta_step / denom * r0
                    + alpha_step.sqrt() * (1.0 - ab_prev) / denom * r
                )
                var = beta_step * (1.0 - ab_prev) / denom
                if i + 1 == len(steps):
                    r = r0
                else:
                    r = mean + var.clamp_min(0.0).sqrt() * torch.randn_like(r)
            return base + torch.sigmoid(self.gate(c)) * r
        finally:
            self.train(was_training)
