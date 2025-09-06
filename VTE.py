#copyright joshuah.rainstar@gmail.com 2025
#GPT framework from karapathy et al
#various ideas and concepts annotated as i get to them
from __future__ import annotations
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple

class Cell(nn.Module):
    def __init__(self, dim_in: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden, bias=False) #dont change, false intentional
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(hidden, dim_in, bias=True)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))   

class RecurrentMLP(nn.Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.k = 2 #can set to 3, but marginal gains
        self.hidden = dim_in*2 #if overfitting reduce to dim_in or even dim_in//2
        self.cells_a = nn.ModuleList([Cell(dim_in, self.hidden) for _ in range(self.k)])
    def forward(self, x):
        z = x
        for i in range(self.k):
            z = z + self.cells_a[i](z)
        return z


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple,List
import math
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sparse-gradient router; gradients only to chosen k logits
class RouterTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, k, tau):
        topv, topi = torch.topk(z, k, dim=1, largest=True, sorted=False)
        w = torch.softmax(topv / (tau + 1e-8), dim=1)
        ctx.save_for_backward(topi, w)
        ctx.z_shape = z.shape
        ctx.tau = float(tau)
        return topi, w

    @staticmethod
    def backward(ctx, grad_topi, grad_w):
        topi, w = ctx.saved_tensors
        tau = ctx.tau
        grad_z = None
        if grad_w is not None:
            s = (grad_w * w).sum(dim=1, keepdim=True)
            grad_topv = (w * (grad_w - s)) / (tau + 1e-8)
            grad_z = torch.zeros(ctx.z_shape, device=w.device, dtype=w.dtype)
            grad_z.scatter_add_(1, topi, grad_topv)
        return grad_z, None, None

def router_topk(z, k, tau):
    return RouterTopK.apply(z, k, tau)

def _apply_mixture_grouped(x_flat, topi, weights, W):
    """
    Group tokens by expert and do large GEMMs:
    x_flat:  [N, in]
    topi:    [N, k] (long)
    weights: [N, k]
    W:       [L, out, in]
    return:  [N, out]
    """
    N, in_dim = x_flat.shape
    L, out_dim, in_dim_w = W.shape
    assert in_dim == in_dim_w

    # Flatten assignments
    k = topi.shape[1]
    token_idx = torch.arange(N, device=topi.device).repeat_interleave(k)        # [N*k]
    expert_idx = topi.reshape(-1)                                               # [N*k]
    w_flat = weights.reshape(-1)                                                # [N*k]

    # Sort by expert to form contiguous segments
    expert_sorted, order = torch.sort(expert_idx)
    token_sorted = token_idx.index_select(0, order)
    w_sorted = w_flat.index_select(0, order)

    # Segment boundaries for each used expert
    changed = torch.ones_like(expert_sorted, dtype=torch.bool)
    changed[1:] = expert_sorted[1:] != expert_sorted[:-1]
    seg_starts = torch.nonzero(changed, as_tuple=False).flatten()
    seg_ends = torch.empty_like(seg_starts)
    seg_ends[:-1] = seg_starts[1:]
    seg_ends[-1] = expert_sorted.numel()
    used_experts = expert_sorted.index_select(0, seg_starts)

    y = x_flat.new_zeros(N, out_dim)

    # Loop per used expert; usually small (<= L) and fast vs GEMM cost
    for exp, s, e in zip(used_experts.tolist(), seg_starts.tolist(), seg_ends.tolist()):
        idx = token_sorted[s:e]                        # [m]
        ws = w_sorted[s:e].unsqueeze(1)               # [m, 1]
        X = x_flat.index_select(0, idx)               # [m, in]
        # y_e = X @ W[exp].T  (F.linear is faster/cleaner for row-major)
        y_e = F.linear(X, W[exp])                      # [m, out]
        y_e.mul_(ws)                                   # scale by gate weights (cheap: [m, out])
        y.index_add_(0, idx, y_e)                      # scatter-accumulate
    return y


def _apply_bias_grouped(topi, weights, B):
    """
    Bias via grouped adds (no big GEMM):
    topi:    [N, k] (long)
    weights: [N, k]
    B:       [L, out]
    return:  [N, out]
    """
    N, k = topi.shape
    out_dim = B.shape[1]

    token_idx = torch.arange(N, device=topi.device).repeat_interleave(k)        # [N*k]
    expert_idx = topi.reshape(-1)                                               # [N*k]
    w_flat = weights.reshape(-1)                                                # [N*k]

    expert_sorted, order = torch.sort(expert_idx)
    token_sorted = token_idx.index_select(0, order)
    w_sorted = w_flat.index_select(0, order)

    changed = torch.ones_like(expert_sorted, dtype=torch.bool)
    changed[1:] = expert_sorted[1:] != expert_sorted[:-1]
    seg_starts = torch.nonzero(changed, as_tuple=False).flatten()
    seg_ends = torch.empty_like(seg_starts)
    seg_ends[:-1] = seg_starts[1:]
    seg_ends[-1] = expert_sorted.numel()
    used_experts = expert_sorted.index_select(0, seg_starts)

    y = B.new_zeros((N, out_dim))
    for exp, s, e in zip(used_experts.tolist(), seg_starts.tolist(), seg_ends.tolist()):
        idx = token_sorted[s:e]                 # [m]
        ws = w_sorted[s:e].unsqueeze(1)        # [m, 1]
        y.index_add_(0, idx, ws * B[exp].unsqueeze(0))  # [m, out]
    return y

class FastLearnedCellX3(nn.Module):
    def __init__(self, D_in, H, D_out,
                 L_w1=12, L_w2=12, L_b2=12,
                 k1=3, k2=3, k3=3, tau1=1.0, tau2=1.0, tau3=1.0,
                 d_addr=64,
                 learn_addr=False,
                 learn_tape_w1=True, learn_tape_w2=True, learn_tape_b2=True):
        super().__init__()
        self.D_in, self.H, self.D_out = int(D_in), int(H), int(D_out)
        self.L_w1, self.L_w2, self.L_b2 = int(L_w1), int(L_w2), int(L_b2)
        self.k1, self.k2, self.k3 = int(k1), int(k2), int(k3)
        self.t1, self.t2, self.t3 = float(tau1), float(tau2), float(tau3)

        self.P = nn.Linear(self.D_in, int(d_addr), bias=False)
        if not learn_addr:
            for p in self.P.parameters():
                p.requires_grad = False
            with torch.no_grad():
                nn.init.normal_(self.P.weight, std=1.0 / math.sqrt(self.D_in))

        def init_U(L, d, learn):
            U = torch.randn(L, d)
            U = U - U.mean(dim=1, keepdim=True)
            U = U / (U.norm(dim=1, keepdim=True) + 1e-8)
            return nn.Parameter(U, requires_grad=learn)

        d_addr = int(d_addr)
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # Make params contiguous; channels-last-ish helps matmul kernels
        self.W1 = nn.Parameter(
            F.normalize(torch.randn(self.L_w1, self.H, self.D_in), dim=(1, 2)).contiguous(),
            requires_grad=learn_tape_w1
        )
        self.W2 = nn.Parameter(
            F.normalize(torch.randn(self.L_w2, self.D_out, self.H), dim=(1, 2)).contiguous(),
            requires_grad=learn_tape_w2
        )
        self.b2 = nn.Parameter(
            F.normalize(torch.randn(self.L_b2, self.D_out), dim=1).contiguous(),
            requires_grad=learn_tape_b2
        )

        self.act = nn.GELU()

    def _address(self, x_addr: torch.Tensor):
        U_pack = torch.cat([self.U1, self.U2, self.U3], dim=0)          # [Ltot, d]
        Z = x_addr @ U_pack.t()                                         # [N, Ltot]
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        z1, z2, z3 = torch.split(Z, (s1, s2, s3), dim=1)

        i1, w1 = router_topk(z1, self.k1, self.t1)
        i2, w2 = router_topk(z2, self.k2, self.t2)
        i3, w3 = router_topk(z3, self.k3, self.t3)
        return (i1, w1), (i2, w2), (i3, w3)

    def forward(self, x):
        if x.ndim == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            B, T = x.shape[0], 1
            x_flat = x

        x_addr = self.P(x_flat)                                        # [N, d_addr]
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)

        h = _apply_mixture_grouped(x_flat, i1, w1, self.W1)  # [N, H]
        h = self.act(h)

        
        y = _apply_mixture_grouped(h, i2, w2, self.W2)       # [N, D_out]
        y = y + _apply_bias_grouped(i3, w3, self.b2)
      

        return y.view(B, T, self.D_out) if x.ndim == 3 else y



        
class PairwiseRotSpiral(nn.Module):
    def __init__(self, dim, radius=6.0, omega=1.0, k=1.0, step=0.1, cube_shell=False):
        super().__init__()
        self.dim = dim
        self.radius = float(radius)
        self.omega = float(omega)
        self.k = float(k)
        self.step = float(step)
        self.cube_shell = bool(cube_shell)
        self.eps = 1e-8

    def _cos_sin(self, x):
        theta = self.omega * self.step
        # Use Python math for scalar, then create tensors on correct device and dtype
        c = torch.tensor(math.cos(theta), device=x.device, dtype=x.dtype)
        s = torch.tensor(math.sin(theta), device=x.device, dtype=x.dtype)
        return c, s

    def forward(self, x):
        D = x.size(-1)
        # radial term
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        radial = (self.radius - r) * (x / r)

        # rotation on 2D pairs, vectorized
        if D >= 2:
            c, s = self._cos_sin(x)
            n2 = D // 2
            head = x[..., : n2 * 2].reshape(*x.shape[:-1], n2, 2)
            xi = head[..., 0]
            xj = head[..., 1]
            yi = c * xi - s * xj
            yj = s * xi + c * xj
            rot = torch.stack([yi, yj], dim=-1).reshape(*x.shape[:-1], n2 * 2)
            if D % 2 == 1:
                y = torch.cat([rot, x[..., -1:].contiguous()], dim=-1)
            else:
                y = rot
        else:
            y = x

        # one-step Euler update
        y = x + self.step * ((y - x) + self.k * radial)

        if self.cube_shell:
            y = self.radius * torch.tanh(y / self.radius)
        return y



# Example mixer that spirals EACH component around a chosen center (origin by default)
class SpiralMix(nn.Module):
    def __init__(self, rank, **spiral_kwargs):
        super().__init__()
        self.rank = rank
        self.flow = PairwiseRotSpiral(rank, **spiral_kwargs)

    def forward(self, comps, center=None, loop_iters=2):
        # Accept either a list/tuple of [...,] Tensors or a single Tensor [..., r]
        if isinstance(comps, (list, tuple)):
            # old DynMix API: list of [B,T] or [B] -> stack on last dim -> [B,T,r] (or [B,r])
            x = torch.stack(comps, dim=-1)
            return_list = True
        else:
            # new API: comps is already [B,T,r] (or any leading dims, last is r)
            x = comps
            return_list = False

        if center is None:
            center = 0.0  # broadcastable scalar OK
        y = x
        for _ in range(loop_iters):
            y = self.flow(y - center) + center  # pairwise rotations on last dim only

        if return_list:
            # match DynMix return type: list of [...,] tensors
            return [y[..., i] for i in range(y.size(-1))]
        return y

        
class PhaseTap(nn.Module):
    """
    Phase-preserving vector shift with guarded Householder.
    x: (B,T,C) -> y: (B,T,C)
      - t < d:  y[:, t, :] = (1/(d - t)) * a
      - t >= d: y[:, t, :] = H(x_t)^T @ (x_t - x_{t-d})
    Guards:
      - near u_t ≈ a: skip reflection, use identity on v
      - near u_t ≈ -a: use fixed orthonormal b
      - near zero ||x_t||: skip reflection
    """
    def __init__(self, d: int, tau: float = 1e-6):  # ?1 tau
        super().__init__()
        assert isinstance(d, int) and d >= 1
        self.d = d
        self.tau = float(tau)

    @staticmethod
    def _norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.linalg.vector_norm(v, dim=-1).clamp_min(eps)

    @staticmethod
    def _safe_unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        n = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
        return v / n

    def _apply_householder_sym(self, a: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply H v with H a = u, H = I - 2 w w^T, symmetric so H^T = H.
        a,u,v: (..., C)
        Guards near a, near -a, and near zero u.
        """
        C = a.shape[-1]
        # masks
        dot = (a * u).sum(dim=-1, keepdim=True)                      # (...,1)
        near_pos = (dot > 1.0 - self.tau).squeeze(-1)                # (...)
        near_neg = (dot < -1.0 + self.tau).squeeze(-1)               # (...)
        near_zero_u = (torch.linalg.vector_norm(u, dim=-1) < self.tau)  # (...)

        y = v.clone()

        # general case mask
        gen = ~(near_pos | near_neg | near_zero_u)
        if gen.any():
            w = self._safe_unit(a[gen] - u[gen])
            wTv = (w * v[gen]).sum(dim=-1, keepdim=True)
            y[gen] = v[gen] - 2.0 * w * wTv

        # near -a: reflect across fixed b orthonormal to a
        if near_neg.any():
            if C == 1:
                y[near_neg] = -v[near_neg]
            else:
                b = torch.zeros_like(a[near_neg])
                b[..., 1] = 1.0
                bbT_v = (b * v[near_neg]).sum(dim=-1, keepdim=True)
                y[near_neg] = v[near_neg] - 2.0 * b * bbT_v

        # near +a or near zero u: identity on v
        # y[near_pos] and y[near_zero_u] already equal v by init

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "x must be (B,T,C)"
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        y = torch.zeros_like(x)

        # anchor a = e0
        a = torch.zeros(B, 1, C, device=device, dtype=dtype)
        a[..., 0] = 1.0

        # early baseline
        if self.d > 0:
            t_idx = torch.arange(T, device=device)
            early_mask = t_idx < self.d
            if early_mask.any():
                denom = (self.d - t_idx[early_mask]).to(dtype=dtype)
                y[:, early_mask, :] = a.expand(B, early_mask.sum(), C) * denom.unsqueeze(0).reciprocal().unsqueeze(-1)

        if T <= self.d:
            return y

        # main region
        x_t  = x[:, self.d:, :]          # (B,T-d,C)
        x_tm = x[:, :-self.d, :]         # (B,T-d,C)
        u_t  = self._safe_unit(x_t)      # (B,T-d,C)

        a_bt = a.expand(B, x_t.shape[1], C)
        v    = x_t - x_tm

        if C == 1:
            y[:, self.d:, :] = v
            return y

        y[:, self.d:, :] = self._apply_householder_sym(a_bt, u_t, v)
        return y
        
class PhaseTransport(nn.Module):
    def __init__(self, d: int, tau: float = 1e-6):
        super().__init__()
        assert isinstance(d, int) and d >= 1
        self.d = d
        self.tau = float(tau)

    @staticmethod
    def _norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.linalg.vector_norm(v, dim=-1).clamp_min(eps)

    @staticmethod
    def _safe_unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        n = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
        return v / n

    @staticmethod
    def _orthonormal_perp(v: torch.Tensor) -> torch.Tensor:
        # v: (N, C) assumed nonzero
        N, C = v.shape
        idx = torch.argmin(torch.abs(v), dim=-1)      # pick coord with smallest magnitude
        e = torch.zeros_like(v)
        e.scatter_(1, idx.unsqueeze(1), 1.0)
        p = e - (e * v).sum(dim=-1, keepdim=True) * v # Gram-Schmidt
        p = p / PhaseTransport._norm(p).unsqueeze(-1)
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "x must be (B,T,C)"
        B, T, C = x.shape
        device, dtype = x.device, x.dtype
        y = torch.zeros_like(x)

        # early baseline with per-sequence direction, not a global axis
        if T > 0:
            ref_t = min(self.d, T - 1)
            u_ref = self._safe_unit(x[:, ref_t, :])  # (B, C)
            if self.d > 0:
                t_idx = torch.arange(T, device=device)
                early_mask = t_idx < self.d
                if early_mask.any():
                    denom = (self.d - t_idx[early_mask]).to(dtype=dtype)     # (Te,)
                    scales = (1.0 / denom).view(1, -1, 1)                    # (1, Te, 1)
                    y[:, early_mask, :] = u_ref.view(B, 1, C) * scales       # (B, Te, C)

        if T <= self.d:
            return y

        # main region t >= d
        xt  = x[:, self.d:, :]             # (B, T-d, C)
        xtm = x[:, :-self.d, :]            # (B, T-d, C)
        u   = self._safe_unit(xt)          # (B, T-d, C)
        v   = self._safe_unit(xtm)         # (B, T-d, C)
        w   = xt - xtm                      # (B, T-d, C)

        c = (u * v).sum(dim=-1, keepdim=True)          # (B, T-d, 1)
        # squeeze masks to (B, T-d)
        near_pos = (c > 1.0 - self.tau).squeeze(-1)
        near_neg = (c < -1.0 + self.tau).squeeze(-1)
        small_u  = (torch.linalg.vector_norm(xt,  dim=-1) < self.tau)
        small_v  = (torch.linalg.vector_norm(xtm, dim=-1) < self.tau)
        trivial  = near_pos | small_u | small_v

        y_main = w.clone()

        # general case
        gen = ~(trivial | near_neg)
        if gen.any():
            u_g = u[gen]                       # (N, C)
            v_g = v[gen]
            w_g = w[gen]
            c_g = c[gen].unsqueeze(-1)[:, 0, :]  # (N, 1) ensure 2D
            alpha = 1.0 / (1.0 + c_g).clamp(min=self.tau)

            a = (v_g * w_g).sum(dim=-1, keepdim=True)  # v·w
            b = (u_g * w_g).sum(dim=-1, keepdim=True)  # u·w
            Kw  = u_g * a - v_g * b
            K2w = u_g * (a * c_g - b) + v_g * (b * c_g - a)
            y_main[gen] = w_g - Kw + alpha * K2w

        # antipodal 180 deg
        if near_neg.any():
            v_n = v[near_neg]                 # (N, C)
            w_n = w[near_neg]
            p   = self._orthonormal_perp(v_n) # (N, C)
            proj_v = (v_n * w_n).sum(dim=-1, keepdim=True) * v_n
            proj_p = (p   * w_n).sum(dim=-1, keepdim=True) * p
            y_main[near_neg] = w_n - 2.0 * proj_v - 2.0 * proj_p

        y[:, self.d:, :] = y_main
        return y






"""
Manifold Attention (no learned attention) with deterministic subspace iteration.

Core idea
---------
Treat X in [B, T, D] as a curve in R^D over time. Build a compact, self-adjoint
operator C = (1/T) X'^T X' with X' = X - anchor + low_rank_shift(X). Extract a
rank-r invariant subspace with K steps of deterministic subspace iteration.
Project onto that basis to obtain r scalar traces, apply simple analytic
conditioning (energy normalization and optional soft shrinkage, optional causal
AR(1)), then reconstruct with the (orthonormal) basis and undo the shift. No
query-key-value attention, no near/far field.

Notes
-----
- Subspace iteration is deterministic and differentiable. We use batched QR to
  orthonormalize after each step. K controls the number of power iterations.
  If you want K to play the role of "heads", think of each iteration as a head
  that sharpens alignment to the top-r invariant subspace. In practice we use
  the final V_K for projection.
- Low-rank shift S(X) = U sigma(V^T X) is optional and helps undo harmful
  normalization. Set bottleneck "shift_rank" to 0 to disable.
- The basis columns are sign-aligned to the first token so that they are stable
  across steps and batches.
- Reconstruction uses V^T directly since columns are orthonormal. If you swap
  orthonorm for another routine, you can still use a tiny r x r solve.

"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _batch_eye(n: int, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Batched identity [B, n, n]."""
    I = torch.eye(n, device=device, dtype=dtype)
    return I.unsqueeze(0).expand(batch, n, n)


def orthonorm_columns(V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Orthonormalize columns of V with batched QR.

    V: [B, D, r]  -> returns Q: [B, D, r] with Q^T Q = I_r
    """
    # torch.linalg.qr supports batched input
    Q, R = torch.linalg.qr(V, mode="reduced")
    # Ensure a consistent sign by forcing diag(R) positive where possible
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    sgn = torch.sign(diag + eps).unsqueeze(-2)  # [B, 1, r]
    Q = Q * sgn
    return Q


def subspace_iteration(C: torch.Tensor, r: int, K: int, V0: Optional[torch.Tensor] = None,
                       eps: float = 1e-6) -> torch.Tensor:
    """
    Batched subspace iteration with a Student-t-like spectral filter.
    Same signature and return as before. C: [B, D, D], V_K: [B, D, r].
    """
    B, D, _ = C.shape
    device, dtype = C.device, C.dtype

    # Deterministic init
    if V0 is None:
        E = torch.zeros(D, r, device=device, dtype=dtype)
        E[:r, :r] = torch.eye(r, device=device, dtype=dtype)
        V = E.unsqueeze(0).expand(B, D, r).contiguous()
    else:
        V = V0

    # Build block-Krylov basis: Q = [V, CV, C^2V, ...] with K blocks
    blocks = []
    V = orthonorm_columns(V, eps=eps)
    Z = V
    for _ in range(max(1, K)):
        blocks.append(Z)
        Z = torch.matmul(C, Z)
        Z = orthonorm_columns(Z, eps=eps)

    Q = torch.cat(blocks, dim=2)  # [B, D, q], q = r*K
    Q = orthonorm_columns(Q, eps=eps)

    # Small projected matrix H = Q^T C Q  -> shape [B, q, q]
    H = torch.matmul(Q.transpose(1, 2), torch.matmul(C, Q))

    # EVD of H
    evals, U = torch.linalg.eigh(H)  # ascending per batch; evals: [B, q], U: [B, q, q]

    # Student-t-like increasing, saturating filter on eigenvalues
    # Choose scale and df to taste; these are stable defaults.
    # κ: scale, set from a high quantile of evals per batch. ν: degrees of freedom.
    kappa = torch.quantile(evals.clamp_min(eps), 0.80, dim=-1, keepdim=True) + eps
    nu = 4.0  # heavier tails for smaller ν; tune as needed

    gt = 1.0 - torch.pow(1.0 + evals / kappa, -0.5 * nu)   # [B, q], in (0,1)
    # Optional additional tempering to keep order but soften dominance
    # Use fractional power on λ to compress ratios
    p = 0.5
    scores = torch.pow(evals.clamp_min(eps), p) * gt        # [B, q]

    # Pick the r columns of U with largest filtered scores
    idx = scores.argsort(dim=-1, descending=True)[..., :r]  # [B, r]
    idx_exp = idx.unsqueeze(1).expand(B, U.size(1), r)      # [B, q, r]
    U_top = torch.gather(U, 2, idx_exp)                    # [B, q, r]

    # Lift back: V = Q @ U_top, then orthonormalize
    V = torch.matmul(Q, U_top)                              # [B, D, r]
    V = orthonorm_columns(V, eps=eps)
    return V


def sign_align(V: torch.Tensor, a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Align signs of columns of V so that v_i^T a >= 0 for each i.

    V: [B, D, r]
    a: [B, D]  (anchor token x_1)
    returns V with column-wise signs adjusted deterministically.
    """
    # Compute dot products per column: [B, r]
    dots = (V * a.unsqueeze(-1)).sum(dim=1)
    sgn = torch.sign(dots + eps)  # +eps to avoid 0 sign
    return V * sgn.unsqueeze(1)


def energy_normalize(traces: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-component energy normalization over time.

    traces: [B, T, r]
    returns (normed_traces, scales) where scales: [B, 1, r]
    """
    # Energy per component across time
    scales = torch.sqrt(torch.clamp((traces**2).sum(dim=1, keepdim=True), min=0.0) + eps)
    traces_n = traces / scales
    return traces_n, scales


def soft_shrink(x: torch.Tensor, lam: float) -> torch.Tensor:
    if lam <= 0.0:
        return x
    # Elementwise soft threshold
    return torch.sign(x) * F.gelu(torch.abs(x) - lam)


def ar1_filter(x: torch.Tensor, rho: float) -> torch.Tensor:
    """Causal AR(1) smoothing along time dimension for each component independently.

    x: [B, T, r], rho in [0,1)
    returns y of same shape
    """
    if rho <= 0.0:
        return x
    B, T, r = x.shape
    y = torch.zeros_like(x)
    y[:, 0, :] = x[:, 0, :]
    for t in range(1, T):
        y[:, t, :] = rho * y[:, t - 1, :] + (1.0 - rho) * x[:, t, :]
    return y


class LowRankShift(nn.Module):
    """Low-rank residual shift S(X) = U sigma(V^T X) applied per time step.

    If shift_rank == 0, the caller should bypass this module.
    """

    def __init__(self, d_model: int, shift_rank: int):
        super().__init__()
        self.d_model = d_model
        self.shift_rank = shift_rank
        self.in_proj = nn.Linear(d_model, shift_rank, bias=False)
        self.out_proj = nn.Linear(shift_rank, d_model, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        z = self.in_proj(x)
        z = self.act(z)
        s = self.out_proj(z)
        return s


        
def subspace_iteration_linop(matvec, d, rank, K, V0, eps: float = 1e-6):
    """
    Batched subspace iteration using a linear-operator matvec.
    - matvec: function(V) -> M @ V with V [B, d, r] and returns [B, d, r]
    - d: ambient dimension (D)
    - rank: r
    - K: iterations
    - V0: required init [B, d, r] - use the same identity init as covariance path
    """
    V = orthonorm_columns(V0, eps=eps)
    for _ in range(max(1, K)):
        Z = matvec(V)                  # [B, d, r]
        V = orthonorm_columns(Z, eps)  # match covariance path behavior
    return V
    
#https://arxiv.org/pdf/2503.10622
class TanhNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 0.9)
        self.delta = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
    def forward(self, x):
        x = F.tanh(self.scale * x)
        return self.delta * x + self.bias
        
class ManifoldAttentionNoAttn(nn.Module):
    def __init__(
        self,
        config,
        rank: int,
        K: int = 2,
        shift_rank: int = 0,
        shrink_lambda: float = 0.0,
        causal: bool = False,
        ar_rho: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert rank > 0 and K >= 1
        self.d_model = config.n_embd
        self.rank = rank                # <-- fix: respect constructor
        self.K = K
        self.shift_rank = shift_rank
        self.shrink_lambda = float(shrink_lambda)
        self.causal = bool(causal)
        self.ar_rho = float(ar_rho)
        self.eps = float(eps)
        self.ln = TanhNorm(config.n_embd)

        self.shift = LowRankShift(self.d_model, shift_rank) if shift_rank > 0 else None
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.dynmix = SpiralMix(1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> y: [B, T, D]"""
        B, T, D = x.shape
        assert D == self.d_model

        # Anchor vector (no large allocs)
        anchor = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        anchor[:, 0] = 1.0

        # Center
        xc = x - anchor.unsqueeze(1)  # broadcast over T

        # Optional low-rank de-normalization shift; avoid adding zeros if not needed
        if self.shift is not None:
            s = self.shift(x)
            xprime = xc + s
        else:
            s = None
            xprime = xc

        # Shapes
        xt = xprime.transpose(1, 2)  # [B, D, T]

        # Optimized: linear operator form with the SAME init as covariance path
        # Build V0 as first r columns of the identity, expanded over batch
        E = torch.zeros(B, D, self.rank, device=x.device, dtype=x.dtype)
        E[:, :self.rank, :self.rank] = torch.eye(self.rank, device=x.device, dtype=x.dtype)

        def cov_matvec(V):  # V: [B, D, r] -> [B, D, r]
            Y = torch.matmul(xprime, V)           # [B, T, r]
            Z = torch.matmul(xt, Y) / float(T)    # [B, D, r]
            return Z + self.eps * V

        V = subspace_iteration_linop(
            cov_matvec, D, self.rank, self.K, V0=E, eps=self.eps
        )

        # Sign alignment using anchor token
        V = sign_align(V, anchor)  # [B, D, r]

        # Project to r scalar traces over time: [B, T, r]
        traces = torch.matmul(xprime, V)  # [B, T, r]

        # Analytic conditioning
        traces_n, scales = energy_normalize(traces, eps=self.eps)
        traces_n = soft_shrink(traces_n, self.shrink_lambda)

        traces_list = [traces_n[..., i] for i in range(traces_n.size(-1))]
        prod = self.dynmix(traces_list)
        traces_n = torch.stack(prod, dim=-1)

        if self.causal and self.ar_rho > 0.0:
            traces_n = ar1_filter(traces_n, self.ar_rho)

        traces_final = traces_n * scales

        # Recompose
        x_tilde = torch.matmul(traces_final, V.transpose(1, 2))  # [B, T, D]

        # Undo shift and add anchor
        if s is not None:
            x_hat = x_tilde - s + anchor.unsqueeze(1)
        else:
            x_hat = x_tilde + anchor.unsqueeze(1)


        y = self.dropout(self.out(x_hat))
        return self.ln(y)


    
class LocalSelfAttention(nn.Module):
    """
    Near-field attention with learned sinks (gpt-oss style).
    - window_size must be odd. 5 means 2 before, self, 2 after.
    - causal=False -> symmetric window [t-2, t+2]
    - causal=True  -> left window only [t-2, t]
    - learned sinks per head let heads "do nothing" by allocating mass to a null path.
    """
    def __init__(self, config, window_size=5, causal=True, use_sinks=True, sink_init=0.0):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.causal = causal
        self.window_size = window_size
        self.half_w = window_size // 2
        self.use_sinks = use_sinks

        # projections match original attention so you can swap in-place
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # learned per-head sinks, live in logits space
        # shape [n_head], broadcast over batch and time
        if self.use_sinks:
            self.sinks = nn.Parameter(torch.full((self.n_head,), float(sink_init)))
        else:
            self.register_parameter("sinks", None)

        # optional speedup: preallocate buffer indices after you know block_size
        # left as runtime-computed to stay shape-agnostic here

        self.dynmix = SpiralMix(config.n_embd)
        self.ln = TanhNorm(config.n_embd)

    def _build_local_index(self, T, device):
            t_idx = torch.arange(T, device=device)                # (T,)
            offsets = torch.arange(-self.half_w, self.half_w + 1, device=device)  # (W,)
            neigh = t_idx[:, None] + offsets[None, :]             # (T, W)
            valid = (neigh >= 0) & (neigh < T)
            if self.causal:
                valid &= (neigh <= t_idx[:, None])
            neigh_clamped = neigh.clamp(0, T - 1)                 # safe gather
            return neigh_clamped, valid

    def forward(self, x, sinks_override: torch.Tensor | None = None):
        B, T, C = x.size()

        # q, k, v and split heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        s = [q,k,v]
        s = self.dynmix(s)
        q = s[0]
        k = s[1]
        #v= s[2] #do not mix V!
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)

        # local neighborhood indices and validity mask
        neigh_idx, valid = self._build_local_index(T, x.device)       # (T,W), (T,W)
        idx = neigh_idx.view(1, 1, T, self.window_size).expand(B, self.n_head, -1, -1)  # (B,H,T,W)

        # gather local keys and values: (B,H,T,W,Dh)
        k_exp = k.unsqueeze(3).expand(B, self.n_head, T, self.window_size, self.head_dim)
        v_exp = v.unsqueeze(3).expand(B, self.n_head, T, self.window_size, self.head_dim)
        k_neigh = torch.gather(k_exp, 2, idx.unsqueeze(-1).expand_as(k_exp))
        v_neigh = torch.gather(v_exp, 2, idx.unsqueeze(-1).expand_as(v_exp))

        # scaled logits to local window
        inv_sqrt_d = 1.0 / math.sqrt(self.head_dim)
        # use float32 math for stability
        scores = (q.unsqueeze(3).to(torch.float32) * k_neigh.to(torch.float32)).sum(-1) * inv_sqrt_d  # (B,H,T,W)

        # mask invalid positions inside window
        scores = scores.masked_fill(~valid.view(1, 1, T, self.window_size), float("-inf"))

        if self.use_sinks or sinks_override is not None:
            # learned sinks per head, in logits space
            sinks_vec = sinks_override if sinks_override is not None else self.sinks
            # shape to broadcast over (B,H,T,W)
            sinks_b = sinks_vec.view(1, self.n_head, 1, 1).to(scores.dtype)

            # numerically stable normalization with sinks
            # max over window per (B,H,T), then joint max with sink
            m = scores.amax(dim=-1, keepdim=True)                              # (B,H,T,1)
            mj = torch.maximum(m, sinks_b)                                     # (B,H,T,1)
            exp_scores = torch.exp(scores - mj)                                # (B,H,T,W)
            exp_sink = torch.exp(sinks_b - mj)                                 # (B,H,T,1)
            denom = exp_scores.sum(dim=-1, keepdim=True) + exp_sink            # (B,H,T,1)
            att = exp_scores / denom                                           # (B,H,T,W)
        else:
            # standard local softmax
            att = F.softmax(scores, dim=-1)

        att = self.attn_dropout(att)

        # weighted sum of local values: (B,H,T,W) x (B,H,T,W,Dh) -> (B,H,T,Dh)
        y = (att.unsqueeze(-1) * v_neigh).sum(3)

        # merge heads, output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return self.ln(y)


class BlockFast(nn.Module):
    def __init__(self, config,d):
        super().__init__()
        self.encoder =FastLearnedCellX3(D_in=config.n_embd,H=config.n_embd*2,D_out=config.n_embd)
        self.decoder = FastLearnedCellX3(D_in=config.n_embd,H=config.n_embd*2,D_out=config.n_embd)
        self.decoder_attn = FastLearnedCellX3(D_in=config.n_embd,H=config.n_embd*2,D_out=config.n_embd)
        #note- CANNOT share with decoder for far field.
        #projection is different, codebook needs to be different.
        #input should be shared.

        self.attn = LocalSelfAttention(config)
        self.distance  =PhaseTransport(1)
        self.ln = TanhNorm(config.n_embd)
        self.convolve  = ManifoldAttentionNoAttn(
                config, rank=16 if d%2 else 8, K=3,
                shift_rank=12, shrink_lambda=0.01,
                causal=False, ar_rho=0.0, eps=1e-5)
        
    def forward(self, x):
        B,T,C= x.shape
        prod = self.distance(self.ln(x))
        prod = prod + self.encoder(prod)
        prod = prod + self.decoder(self.convolve(prod)) + self.decoder_attn(self.attn(prod))
        x = x + prod

        return x

class FixedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(num_embeddings, embedding_dim, generator=g)
        # row-center and row-normalize so rows are zero-mean, unit-norm
        W = W - W.mean(dim=1, keepdim=True)
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        self.weight = nn.Parameter(W, requires_grad=False)

    def forward(self, idx):
        return self.weight[idx]
        
@dataclass
class VTEConfig:
    block_size: int = 1024
    vocab_size: int = 66 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_embd: int = 128
    n_head:int = 8
    bias: bool = True
    dropout: float = 0.1

        
class VirtualTurboEncabulator(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_embd = config.n_embd

        self.transformer = nn.ModuleDict(dict(
            wte = FixedEmbedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([BlockFast(config,1+i) for i in range(config.n_layer)]),


        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    # ---------- forward ----------
    def forward(self, idx, targets=None, eprint=False):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.wte(idx)
        for i in range(self.config.n_layer):

            x = self.transformer.h[i](x)
         

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
