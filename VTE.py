#copyright joshuah.rainstar@gmail.com 2025
#GPT framework from karapathy et al
#various ideas and concepts annotated as i get to them
#sep 9 2025: two-stage component isolation and mixing pipeline
#intended to automate semantic selection and amplification at all stages
from __future__ import annotations
import math
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple,Optional, List


        


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


class ManifoldAttentionNoAttnStage2(nn.Module):
    def __init__(
        self,
        config,
        d_model: int,
        rank: int,
        K: int = 2,
        shift_rank: int = 0,
        shrink_lambda: float = 0.0,
        causal: bool = False,
        ar_rho: float = 0.0,
        eps: float = 1e-5,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        assert rank > 0 and K >= 1
        self.d_model = d_model
        self.rank = rank                # <-- fix: respect constructor
        self.K = K
        self.shift_rank = self.d_model 
        self.shrink_lambda = float(shrink_lambda)
        self.causal = bool(causal)
        self.ar_rho = float(ar_rho)
        self.eps = float(eps)

        self.shift = LowRankShift(d_model, shift_rank) if shift_rank > 0 else None
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.up = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
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

        # Overlapped triad mixing (stride=2, overlap-by-one), done simultaneously.
        traces_n = self.dynmix(traces_n)                                     # 3 x [B, T, num_triads]

        
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

        # Residual + thin output projection and optional norm
        y = x + self.dropout(self.out(x_hat))
        y = self.ln(y)
        return y



def frft_time(z: torch.Tensor, alpha: float, *, t_min: float = -1.0, t_max: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
    """
    Fractional Fourier transform (FrFT) along the time axis (dim=1), batched & differentiable.

    Args:
        z:     Tensor [..., T, ...]; we assume time is at dim=1 (i.e., [B, T, C] is common).
               Real or complex. Returned dtype is complex64/complex128 accordingly.
        alpha: FrFT order in radians (α ∈ ℝ). α=0 -> identity; α=π/2 -> (unitary) FFT up to a global phase.
        t_min, t_max: define a continuous, centered time grid over which the quadratic phases are drawn.
        eps:   numerical floor to avoid division-by-zero near singular α.

    Returns:
        Same shape as z, complex dtype, FrFT applied along dim=1.
    """
    # Move time to axis 1 for convenience (no copy if already there)
    orig_shape = z.shape
    if z.dim() < 2:
        raise ValueError("Input must have a time dimension at dim=1 (e.g., [B, T, ...]).")
    if z.dtype.is_complex:
        zc = z
    else:
        # Promote to complex for phase ops
        zc = z.to(torch.complex64 if z.dtype == torch.float32 else torch.complex128)

    device = zc.device
    dtype  = zc.dtype
    B_like = zc.shape[0]
    T      = zc.shape[1]
    tail   = zc.shape[2:]

    # Wrap α into (-π, π] to stabilize trigs
    a = ((float(alpha) + math.pi) % (2.0 * math.pi)) - math.pi

    # Handle near-identity and near-π cases explicitly (fast, stable)
    if abs(a) < 1e-6:
        return zc
    if abs(abs(a) - math.pi) < 1e-6:
        # α ≈ π -> time reversal with a global phase
        phase = torch.exp(1j * torch.tensor(math.copysign(math.pi/2, a), device=device, dtype=dtype))
        return phase * torch.flip(zc, dims=[1])

    # Core parameters
    s = 1.0 / max(eps, abs(math.sin(a)))             # |csc α|
    s *= torch.sign(torch.tensor(math.sin(a), device=device, dtype=torch.float32)).item()  # keep sign
    c = math.cos(a) / max(eps, math.sin(a))          # cot α

    # Build centered, continuous time grid t in [t_min, t_max]
    t = torch.linspace(t_min, t_max, T, device=device, dtype=torch.float32)
    t = t.to(zc.real.dtype)  # match precision
    dt = (t_max - t_min) / (T - 1) if T > 1 else torch.tensor(1.0, device=device, dtype=t.dtype)

    # Pre- / post- chirps (broadcast over batch + channels)
    # From the chirp-convolution identity:
    #   U[n] = pref * e^{ iπ (cot α + csc α) t_n^2 } * ( g * h )[n],
    #   where g[k] = x[k] e^{ iπ (cot α + csc α) t_k^2 },  h[m] = e^{ -iπ csc α (m*dt)^2 }.
    gamma_plus = (c + s) * (t**2)            # [T]
    pre_post   = torch.exp(1j * math.pi * gamma_plus).reshape(1, T, *([1] * len(tail)))  # [1,T,1,...]

    g = zc * pre_post                        # [B, T, C...]

    # Build the difference-kernel h over index offsets m = -(T-1)..(T-1)
    m = torch.arange(-(T-1), T, device=device, dtype=t.dtype)   # length 2T-1
    h = torch.exp(-1j * math.pi * s * (m * dt)**2)              # [2T-1]

    # FFT-based linear convolution along T
    L = 1 << (2 * T - 1 - 1).bit_length()    # next power-of-two >= (2T-1)
    # Pad g along time, keeping other dims
    pad_g = torch.nn.functional.pad(g, pad=(0,)* (2*len(tail)) + (0, L - T))  # pad last-in-first-out: (..., T) -> (..., L)
    # Embed h into length-L with center at index 0 for circular -> linear extraction
    h_pad = torch.zeros(L, device=device, dtype=dtype)
    # Place h at indices corresponding to m (negative indices wrap)
    idx = (m % L).long()
    h_pad.scatter_(0, idx, h.to(dtype))

    # FFT multiply (along time axis)
    G = torch.fft.fft(pad_g, n=L, dim=1)
    H = torch.fft.fft(h_pad).reshape(1, L, *([1] * len(tail)))
    conv_full = torch.fft.ifft(G * H, n=L, dim=1)

    # Extract the central T samples for *linear* conv: start at offset (T-1)
    start = T - 1
    conv_center = conv_full[:, start:start+T, ...]

    # Post-chirp and prefactor
    pref = torch.sqrt(torch.tensor(1.0, dtype=dtype, device=device) - 1j * torch.tensor(c, dtype=dtype, device=device))
    out = pref * pre_post * conv_center

    # (Optional) scale by dt for integral-like normalization; comment out if you prefer unscaled energy
    out = out * dt

    return out
    
class ManifoldAttentionNoAttnStage1(nn.Module):
    def __init__(
        self,
        config,
        d_model: int,
        rank: int,
        K: int = 2,
        shift_rank: int = 0,
        shrink_lambda: float = 0.0,
        causal: bool = False,
        ar_rho: float = 0.0,
        eps: float = 1e-5,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        assert rank > 0 and K >= 1
        self.d_model = d_model
        self.rank = rank                # <-- fix: respect constructor
        self.K = K
        self.shift_rank = self.d_model 
        self.shrink_lambda = float(shrink_lambda)
        self.causal = bool(causal)
        self.ar_rho = float(ar_rho)
        self.eps = float(eps)

        self.shift = LowRankShift(d_model, shift_rank) if shift_rank > 0 else None
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.up = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
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

        # Precompute per-batch Omega weights from X'  (no top-k, no learning)
        alphas = torch.linspace(0.15, 2.99, steps=self.rank, device=x.device)  # fixed small grid
        p = 0.5
        eps = 1e-6
        
        
        def Komega_apply(Y, weights):  # Y: [B, T, r]
            out = 0
            for (alpha, w) in weights:                  # w: [B, T]
                Y_a = frft_time(Y, alpha)               # [B, T, r]
                out = out + frft_time(w.unsqueeze(-1) * Y_a, -alpha)
            return (out / len(weights)).real
        
        # Build weights once per forward (data-derived, no params)
        weights = []
        for alpha in alphas:
            X_a = frft_time(xprime, alpha)              # [B, T, D]
            E_a = X_a.abs().pow(2).mean(dim=2)          # [B, T]  (avg over features)
            w_a = (E_a + eps).pow(p)
            w_a = w_a / (w_a.mean(dim=1, keepdim=True) + eps)
            weights.append((alpha, w_a))
        
        # matvec used inside subspace_iteration_linop:
        def cov_matvec(V):                               # V: [B, D, r]
            Y = torch.matmul(xprime, V)                  # [B, T, r]
            Y = Komega_apply(Y, weights)                 # Omega operator on traces
            Z = torch.matmul(xt, Y) / float(T)           # [B, D, r]
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

        traces_n = self.dynmix(traces_n)                            

        
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

        # Residual + thin output projection and optional norm
        y = x + self.dropout(self.out(x_hat))
        y = self.ln(y)
        return y

class Cell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd*2, bias=False) #dont change, false intentional
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(config.n_embd*2, config.n_embd, bias=True)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))
        
class AutoencoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distance  =PhaseTransport(1) 
        self.ln = nn.LayerNorm(config.n_embd)
        self.convolve1  = ManifoldAttentionNoAttnStage1(
                config, d_model=config.n_embd, rank=16, K=3,
                shift_rank=8, shrink_lambda=0.01,
                causal=False, ar_rho=0.0, eps=1e-5, dropout=0.0,
                use_layernorm=True
            )#low frequency patterns
        #self.attn  = LocalSelfAttention(config)
        self.convolve2  = ManifoldAttentionNoAttnStage2(
                config, d_model=config.n_embd, rank=16, K=2,
                shift_rank=8, shrink_lambda=0.01,
                causal=False, ar_rho=0.0, eps=1e-5, dropout=0.0,
                use_layernorm=True
            )#higher frequency patterns
        self.enc1 = Cell(config)
        self.dec1 = Cell(config)
    
    def forward(self, x):
        z = self.ln(x)
        z = z + self.distance(z)
        z1 = self.enc1(z)
        z1 = self.convolve2(self.convolve1(z1))
        z1 = self.dec1(z1)
        return x + z1


        
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
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 66 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_embd: int = 128
    n_experts:int = 4
    n_head:int=4
    bias: bool = True
    dropout: float = 0.1

from matplotlib import pyplot as plt
    
        
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_embd = config.n_embd

        self.transformer = nn.ModuleDict(dict(
            wte = FixedEmbedding(config.vocab_size, config.n_embd,seed=123),
            h = nn.ModuleList([AutoencoderBlock(config) for _ in range(config.n_layer)]),

        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    
    # ---------- forward ----------
    def forward(self, idx, targets=None, eprint=False):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.wte(idx) 
        x = x.detach()                 # sever any stale history just in case
        x.requires_grad_(True)         # make x a grad leaf for τ at layer 0
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
