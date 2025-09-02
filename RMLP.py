#copyright joshuah rainstar 2025 joshuah.rainstar@gmail.com
#A simplistic Recurrent MLP offers quite good, unexpected behavior
#it will beat a hard or soft routed MOE, a GRU, etc on many toy tasks. Why?
#note: marginal gain from accumulating multiple RecurrentMLP products.
#95% of work from first RMLP.
#aug 32 2025: tweaks provide incremental gains across diverse problem set.

import torch
import torch.nn as nn
import torch.nn.functional as F
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

#copyright joshuah rainstar 2025 joshuah.rainstar@gmail.com
#A hash-based MOE is also powerful. However, it REALLY complains(takes a lot longer) if there are more than a few experts.
#note- an optimized triton form might not be an issue.
#note: this gets very low loss and runs very quickly and has minimal parameter overhead.

import math
import torch
import torch.nn as nn
from typing import List, Tuple

# ---------------------------
# small number utils
# ---------------------------

def is_prime(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    r = int(n ** 0.5)
    f = 3
    while f <= r:
        if n % f == 0: return False
        f += 2
    return True

def first_primes(k: int, start: int = 3) -> List[int]:
    out, p = [], max(3, start | 1)
    while len(out) < k:
        if is_prime(p): out.append(p)
        p += 2
    return out

def inv_mod(a: int, m: int) -> int:
    # Extended Euclid
    t, new_t, r, new_r = 0, 1, m, a % m
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r
    if r != 1:
        raise ValueError("a not invertible mod m")
    return t % m

def crt_pair(r1: int, m1: int, r2: int, m2: int) -> Tuple[int, int]:
    """
    Solve x ≡ r1 (mod m1), x ≡ r2 (mod m2).
    Returns (x in [0, m1*m2), modulus = m1*m2).
    Assumes m1 and m2 are coprime.
    """
    t = ((r2 - r1) % m2) * inv_mod(m1 % m2, m2) % m2
    x = r1 + m1 * t
    M = m1 * m2
    return x % M, M

# ---------------------------
# modulo hash head
# ---------------------------

class ModuloHash(nn.Module):
    """
    f(x): R^D -> residues r_k in Z_{m_k}
    Steps per channel k:
      s_k = a_k^T x + b_k
      f_k = s_k mod T_k  (fold)
      q_k = round( m_k * f_k / T_k ) mod m_k  (round-off to nearest bin)
    """
    def __init__(
        self,
        D: int,
        moduli: List[int],
        learnable: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.D = D
        self.m = torch.tensor(moduli, dtype=torch.long)  # [K]
        self.K = len(moduli)

        g = torch.Generator().manual_seed(seed)
        W = torch.randn(D, self.K, generator=g) / math.sqrt(D)
        b = torch.randn(self.K, generator=g) * 0.01
        T = torch.ones(self.K)  # periods; start at 1.0
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.register_buffer("T", T)


    def periods(self) -> torch.Tensor:
        return self.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        s = x @ self.W + self.b  # [B, K]
        T = self.periods()       # [K]
        # fold to [0, T)
        f = torch.remainder(s, T)  # [B, K]
        # map to bins [0, m_k)
        m = self.m.to(f.device).to(torch.float32)
        r_float = f * (m / T)                 # [B, K]
        q = torch.floor(r_float + 0.5)        # nearest bin
        q = torch.remainder(q, m)             # wrap edges
        return q.to(torch.long)               # [B, K]

# ---------------------------
# experts
# ---------------------------

class RowWiseExpertsMLP(nn.Module):
    def __init__(self, D: int, H1: int, O: int, E: int):
        super().__init__()
        self.D, self.H1, self.O, self.E = D, H1, O, E
        self.W1 = nn.Parameter(torch.zeros(E, H1, D))
        torch.nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')
        self.W2 = nn.Parameter(torch.zeros(E, O,  H1) )
        torch.nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')
        self.b2 = nn.Parameter(torch.zeros(E, O))

    def forward(self, x: torch.Tensor, eid: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device = x.device
    
        eid_sorted, idx_sort = torch.sort(eid)         # [B]
        x_sorted = x.index_select(0, idx_sort)         # [B, D]
        Y_sorted = torch.empty(B, self.O, device=device, dtype=x.dtype)
    
        for e in range(self.E):
            idx_e = (eid_sorted == e).nonzero(as_tuple=False).squeeze(1)  # [n]
            if idx_e.numel() == 0:
                continue
            X_e = x_sorted.index_select(0, idx_e)                         # [n, D]
            H_e = X_e.matmul(self.W1[e].t())
            H_e = F.gelu(H_e)
            Y_e = H_e.matmul(self.W2[e].t()).add_(self.b2[e])
            Y_sorted.index_copy_(0, idx_e, Y_e)
    
        Y = torch.empty_like(Y_sorted)
        Y[idx_sort] = Y_sorted
        return Y

# ---------------------------
# router with CRT consensus
# ---------------------------

class ModCRTMoE(nn.Module):
    """
    Hard router:
      1) ModuloHash -> residues r_k in Z_{m_k}
      2) Build CRT candidates from all channel pairs
      3) Pick candidate with maximum residue agreement
      4) expert_id = candidate % E
      5) Send raw x to that expert
    """
    def __init__(
        self,
        D: int,
        O: int,
        num_experts: int,
        moduli: List[int] = None,
        expert_hidden: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        self.D, self.O, self.E = D, O, num_experts

        # choose pairwise coprime moduli; default primes until product >= 4E
        if moduli is None:
            K = 3
            while True:
                primes = first_primes(K)
                prod = 1
                for p in primes: prod *= p
                if prod >= max(4 * num_experts, 256):
                    moduli = primes
                    break
                K += 1

        # sanity: pairwise coprime
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if math.gcd(moduli[i], moduli[j]) != 1:
                    raise ValueError("moduli must be pairwise coprime")

        self.moduli = moduli
        self.hash = ModuloHash(D, moduli, seed=seed)

        # precompute pairwise inverses for speed
        K = len(moduli)
        self._pair_idx = []
        self._pair_data = []
        for i in range(K):
            for j in range(i + 1, K):
                m1, m2 = moduli[i], moduli[j]
                inv = inv_mod(m1 % m2, m2)
                self._pair_idx.append((i, j))
                self._pair_data.append((m1, m2, inv))

        # experts
        self.experts = RowWiseExpertsMLP(D, expert_hidden, O,num_experts)

    @torch.no_grad()
    def _crt_pair_batched(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        r: [B, K] residues
        returns:
          cand: [B, P] candidate integers
          modP: [P] corresponding moduli products
        """
        B, K = r.shape
        P = len(self._pair_idx)
        cand = torch.empty(B, P, dtype=torch.long, device=r.device)
        modP = torch.empty(P, dtype=torch.long, device=r.device)
        for p, ((i, j), (m1, m2, inv)) in enumerate(zip(self._pair_idx, self._pair_data)):
            r1 = r[:, i]
            r2 = r[:, j]
            # t = ((r2 - r1) mod m2) * inv mod m2
            t = ((r2 - r1) % m2) * inv % m2
            x = r1 + t * m1
            cand[:, p] = x % (m1 * m2)
            modP[p] = m1 * m2
        return cand, modP

    @torch.no_grad()
    def _consensus_pick(self, r: torch.Tensor, cand: torch.Tensor, modP: torch.Tensor) -> torch.Tensor:
        """
        r: [B, K] residues
        cand: [B, P] candidate integers
        modP: [P]
        returns: best candidate per row in [0, inf), then reduced mod E
        """
        B, K = r.shape
        P = cand.shape[1]
        m = torch.tensor(self.moduli, dtype=torch.long, device=r.device)  # [K]
        # expand for vectorized residue checks
        # For each candidate c and each channel k, check if c % m_k == r_k
        c_exp = cand.unsqueeze(-1)  # [B, P, 1]
        m_exp = m.view(1, 1, K)     # [1, 1, K]
        r_exp = r.unsqueeze(1)      # [B, 1, K]
        match = (c_exp % m_exp) == r_exp  # [B, P, K]
        scores = match.sum(dim=-1)        # [B, P]
        # pick argmax
        best_idx = torch.argmax(scores, dim=1)  # [B]
        best = cand[torch.arange(B, device=r.device), best_idx]  # [B]
        return best % self.E

    @torch.no_grad()
    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          expert_ids: [B]
          residues: [B, K]
        """
        residues = self.hash(x)           # [B, K]
        cand, modP = self._crt_pair_batched(residues)
        expert_ids = self._consensus_pick(residues, cand, modP)
        return expert_ids, residues

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            eid, _ = self.route(x)
        return self.experts(x, eid)


#despite the added complexity, the UltraMemory layer may be more optimal than the hashMOE.
#it offers very low loss.

import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Norm + FFN blocks
# -------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim: int, inner_multiple: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * inner_multiple)
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


@dataclass
class UltraMemv5SharedCfg:
    hidden_size: int
    n_keys: int                 # N
    key_dim: int                # Dk
    tucker_rank: int            # r
    Rb: int                     # value code dim
    Rp: int                     # pre-value code dim
    Qr: int = 32                # row embedding dim (codebook factorization)
    Qc: int = 32                # col embedding dim (codebook factorization)
    ks_S: int = 4               # sparsity for S rows (top-k kept)
    ks_T: int = 4               # sparsity for T rows (top-k kept)
    projector_rank: int = 8     # shared low-rank projector rank

class UltraMemv5Shared(nn.Module):
    def __init__(self, cfg: UltraMemv5SharedCfg):
        super().__init__()
        self.cfg = cfg
        H, N, Dk, r, Rb, Rp = cfg.hidden_size, cfg.n_keys, cfg.key_dim, cfg.tucker_rank, cfg.Rb, cfg.Rp

        # Keys (shared across layers)
        self.K_row = nn.Parameter(torch.randn(r, N, Dk) / math.sqrt(Dk))
        self.K_col = nn.Parameter(torch.randn(r, N, Dk) / math.sqrt(Dk))
        self.core  = nn.Parameter(torch.randn(r, r) / math.sqrt(max(1, r)))  # rank mixing in grid scoring

        # Learned rank mixers for preselect (kills per-forward SVD)
        self.row_mix = nn.Parameter(torch.randn(r))
        self.col_mix = nn.Parameter(torch.randn(r))

        # Factored codebook via row/col embeddings + bilinear heads (no N^2 tables)
        self.row_emb = nn.Embedding(N, cfg.Qr)
        self.col_emb = nn.Embedding(N, cfg.Qc)
        nn.init.normal_(self.row_emb.weight, std=0.01)
        nn.init.normal_(self.col_emb.weight, std=0.01)

        self.row_to_S = nn.Linear(cfg.Qr, Rb, bias=False)
        self.col_to_S = nn.Linear(cfg.Qc, Rb, bias=False)
        self.row_to_T = nn.Linear(cfg.Qr, Rp, bias=False)
        self.col_to_T = nn.Linear(cfg.Qc, Rp, bias=False)
        nn.init.normal_(self.row_to_S.weight, std=0.02)
        nn.init.normal_(self.col_to_S.weight, std=0.02)
        nn.init.normal_(self.row_to_T.weight, std=0.02)
        nn.init.normal_(self.col_to_T.weight, std=0.02)

        # Basis matrices (shared projection heads)
        self.B = nn.Parameter(torch.randn(Rb, H) * (1.0 / math.sqrt(H)))
        self.U = nn.Parameter(torch.randn(Rp, H) * (1.0 / math.sqrt(H)))
        with torch.no_grad():
            d = min(Rb, H)
            self.B[:d, :d] += torch.eye(d)

        # Shared x -> U-space feature
        self.x_to_U = nn.Linear(H, Rp, bias=False)

        
        self.register_buffer("KrfT_cache", torch.empty(0), persistent=False)
        self.register_buffer("KcfT_cache", torch.empty(0), persistent=False)
        self._K_row_ver = -1
        self._K_col_ver = -1

    @torch.no_grad()
    def get_preselect_banks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        N, r, Dk = self.cfg.n_keys, self.cfg.tucker_rank, self.cfg.key_dim

        # Rebuild when version changes or cache is empty or shape changed
        need_row = (
            self.KrfT_cache.numel() == 0
            or self._K_row_ver != self.K_row._version
            or self.KrfT_cache.shape != (r * Dk, N)
        )
        need_col = (
            self.KcfT_cache.numel() == 0
            or self._K_col_ver != self.K_col._version
            or self.KcfT_cache.shape != (r * Dk, N)
        )

        if need_row:
            # permute to [N, r, Dk], then flatten to [N, r*Dk], then transpose to [r*Dk, N]
            KrfT = self.K_row.permute(1, 0, 2).reshape(N, r * Dk).transpose(0, 1).contiguous()
            self.KrfT_cache = KrfT
            self._K_row_ver = self.K_row._version

        if need_col:
            KcfT = self.K_col.permute(1, 0, 2).reshape(N, r * Dk).transpose(0, 1).contiguous()
            self.KcfT_cache = KcfT
            self._K_col_ver = self.K_col._version

        return self.KrfT_cache, self.KcfT_cache
# -------------------------------
# UltraMemv4 layer
#   - Fold r-mixing into single matmul for preselect (no SVD)
#   - Use factored codebook via embeddings + bilinear
#   - Shared projections
# -------------------------------

@dataclass
class UltraMemv5LayerCfg:
    hidden_size: int
    topk_rows: int
    topk_cols: int
    top_m: int
    softmax_tau: float = 1.0

class UltraMemv5Layer(nn.Module):
    def __init__(self, shared: UltraMemv5Shared, layer_cfg: UltraMemv5LayerCfg):
        super().__init__()
        self.S = shared
        self.C = layer_cfg
        H, r, Dk = self.S.cfg.hidden_size, self.S.cfg.tucker_rank, self.S.cfg.key_dim

        # per-layer queries (kept per-layer)
        self.q = nn.Linear(H, 2*r*Dk, bias=False)

        # per-layer near-identity projector
        pr = shared.cfg.projector_rank
        self.Vproj = nn.Linear(shared.cfg.hidden_size, pr, bias=False)
        self.Uproj = nn.Linear(pr, shared.cfg.hidden_size, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _gather_3d_last(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x: [B, R, N], idx: [B, K] -> [B, R, K]
        B, R, N = x.shape
        idx_r = idx.unsqueeze(1).expand(B, R, idx.size(1))
        return torch.gather(x, dim=2, index=idx_r)

    @staticmethod
    def _gather_2d_last(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x: [B, N], idx: [B, K] -> [B, K]
        return torch.gather(x, dim=1, index=idx)

    @staticmethod
    def _topk_row_sparsify(mat: torch.Tensor, k: int) -> torch.Tensor:
        # keep top-k magnitude per row of last dim (no scatter mask)
        if k <= 0 or k >= mat.size(-1):
            return mat
        vals, _ = torch.topk(mat.abs(), k=k, dim=-1)
        thresh = vals[..., -1:].detach()
        return torch.where(mat.abs() >= thresh, mat, torch.zeros_like(mat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, C = self.S, self.C
        H, N, r, Dk = S.cfg.hidden_size, S.cfg.n_keys, S.cfg.tucker_rank, S.cfg.key_dim
        Bsz = x.size(0)

        q_all = self.q(x).view(Bsz, 2, r, Dk)
        qrow = q_all[:, 0]  # [B,r,Dk]
        qcol = q_all[:, 1]  # [B,r,Dk]

        # --------- Preselect rows/cols with single matmul on flattened vectors
        # Prepare flattened key banks: [N, r*Dk]
        # Cached banks are [r*Dk, N], no grad needed for preselect
        KrfT, KcfT = S.get_preselect_banks()  # [r*Dk, N] each

        # fold row/col mixers into the banks once
        sr = S.row_mix.repeat_interleave(Dk).unsqueeze(1)               # [r*Dk,1]
        sc = S.col_mix.repeat_interleave(Dk).unsqueeze(1)               # [r*Dk,1]
        KrfT_eff = KrfT * sr                                            # [r*Dk,N]
        KcfT_eff = KcfT * sc                                            # [r*Dk,N]
        
        # queries without per-batch mixer scale
        qrow_flat = qrow.reshape(Bsz, -1)
        qcol_flat = qcol.reshape(Bsz, -1)
        row_score = qrow_flat @ KrfT_eff
        col_score = qcol_flat @ KcfT_eff
        # Top-k indices
        row_idx = torch.topk(row_score, k=C.topk_rows, dim=1).indices   # [B,Pr]
        col_idx = torch.topk(col_score, k=C.topk_cols, dim=1).indices   # [B,Pc]

        # --------- Build A, Bc for selected rows/cols (for grid scoring)
        # Gather keys per batch selection (avoid computing over all N)
        # K_row: [r, N, Dk] -> expand to [B, r, N, Dk] for batched gather on N
        K_row_b = S.K_row.unsqueeze(0).expand(Bsz, -1, -1, -1)  # [B,r,N,Dk]
        K_col_b = S.K_col.unsqueeze(0).expand(Bsz, -1, -1, -1)  # [B,r,N,Dk]
        
        # Build index tensors to gather [B,r,Pr,Dk] and [B,r,Pc,Dk]
        idx_r = row_idx.unsqueeze(1).unsqueeze(-1).expand(Bsz, S.cfg.tucker_rank, C.topk_rows, Dk)
        idx_c = col_idx.unsqueeze(1).unsqueeze(-1).expand(Bsz, S.cfg.tucker_rank, C.topk_cols, Dk)
        
        K_row_sel = torch.gather(K_row_b, dim=2, index=idx_r)   # [B,r,Pr,Dk]
        K_col_sel = torch.gather(K_col_b, dim=2, index=idx_c)   # [B,r,Pc,Dk]
        
        # Contract with qrow/qcol only on the selected keys
        A_sel = torch.einsum('brpk,brk->brp', K_row_sel, qrow)  # [B,r,Pr]
        B_sel = torch.einsum('brqk,brk->brq', K_col_sel, qcol)  # [B,r,Pc]

        # mix ranks into qrow once (replaces the later core contraction)
        qrow_mixed = torch.einsum('ij,brk->bjk', self.S.core.T, qrow)  # [B,r,Dk]
        
        # use the mixed qrow to form A_sel
        A_sel = torch.einsum('brpk,bjk->bjp', K_row_sel, qrow_mixed)    # [B,r,Pr]
        B_sel = torch.einsum('brqk,brk->brq', K_col_sel, qcol)          # [B,r,Pc]
        
        # single contraction over rank now
        Sgrid = torch.einsum('brp,brn->bpn', A_sel, B_sel)              # [B,Pr,Pc]

        # Select top_m across grid
        B_, Pr, Pc = Sgrid.shape
        S_flat = Sgrid.reshape(B_, Pr * Pc)
        top_scores, top_idx = torch.topk(S_flat, k=self.C.top_m, dim=1)
        row_pick = torch.div(top_idx, Pc, rounding_mode='trunc')  # [B,M]
        col_pick = top_idx % Pc                                   # [B,M]
        picked_rows = self._gather_2d_last(row_idx, row_pick)     # [B,M]
        picked_cols = self._gather_2d_last(col_idx, col_pick)     # [B,M]

        # Weights with temperature
        if C.softmax_tau != 0:
            weights = F.softmax(top_scores / C.softmax_tau, dim=1)    # [B,M]
        else:
            weights = top_scores

        # --------- Bilinear factored codebook lookups via embeddings
        # row/col embeddings for selected pairs
        row_vecs = S.row_emb(picked_rows.view(-1))  # [B*M, Qr]
        col_vecs = S.col_emb(picked_cols.view(-1))  # [B*M, Qc]

        S_rows = (S.row_to_S(row_vecs) + S.col_to_S(col_vecs)).view(Bsz, self.C.top_m, S.cfg.Rb)
        T_rows = (S.row_to_T(row_vecs) + S.col_to_T(col_vecs)).view(Bsz, self.C.top_m, S.cfg.Rp)

        # Sparsify rows
        if S.cfg.ks_S > 0:
            S_rows = self._topk_row_sparsify(S_rows, S.cfg.ks_S)
        if S.cfg.ks_T > 0:
            T_rows = self._topk_row_sparsify(T_rows, S.cfg.ks_T)

        # --------- Pre-value feature and accumulation
        u = S.x_to_U(x)  # [B,Rp]
        pv = torch.bmm(T_rows, u.unsqueeze(-1)).squeeze(-1)  # [B,M]
        a = weights * pv                             # [B,M]

        # accumulate in Rb then expand with B
        s_acc = (a.unsqueeze(-1) * S_rows).sum(dim=1)                 # [B,Rb]
        s_acc = torch.nn.functional.normalize(s_acc, p=2, dim=-1)     # bound update
        G = s_acc @ S.B                                               # [B,H]

        # Shared near-identity projector
        low_rank = self.Uproj(self.Vproj(G))                          # [B,H]
        out = G + torch.tanh(self.gamma) * low_rank
        return out

# -------------------------------
# Transformer block with UltraMemv4
# -------------------------------

class TransformerBlockUltraMemv5(nn.Module):
    def __init__(self, shared: UltraMemv5Shared, layer_cfg: UltraMemv5LayerCfg, ffn_multiple: float = 2.0, dropout: float = 0.0):
        super().__init__()
        H = shared.cfg.hidden_size
        self.norm_ffn = RMSNorm(H)
        self.norm_mem = RMSNorm(H)
        self.ffn = FeedForward(H, inner_multiple=ffn_multiple, dropout=dropout)
        self.mem = UltraMemv5Layer(shared, layer_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel residuals
        ffn_out = self.ffn(self.norm_ffn(x))
        mem_out = self.mem(self.norm_mem(x))
        return x + ffn_out + mem_out

# -------------------------------
# UltraMemv4 classifier (stack of blocks)
# -------------------------------

class UltraMemv5Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        n_blocks: int,
        shared: UltraMemv5Shared,
        layer_cfg: UltraMemv5LayerCfg,
        ffn_multiple: float = 2.0,
        num_classes: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Identity() if input_dim == hidden_size else nn.Linear(input_dim, hidden_size, bias=False)
        self.blocks = nn.ModuleList([
            TransformerBlockUltraMemv5(shared, layer_cfg, ffn_multiple=ffn_multiple, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        for blk in self.blocks:
            z = blk(z)
        z = self.final_norm(z)
        return self.head(z)

# -------------------------------
# Example: small UltraMemv4 config + training call
# (assumes your Block 1 setup is already in the notebook:
#  - X_train / train_loader / val_loader
#  - train_model(...)
#  - DEVICE, CFG, etc.)
# -------------------------------

H = X_train.shape[1]                      # input dim from your dataset
N = 64                                    # number of row/col keys
Dk = 32                                   # key dim
r = 1                                     # tucker rank
Rb = 32                                   # value code basis size
Rp = 32                                   # pre-value code basis size
ks = 4                                    # sparsity per code row
topk_rows = 16
topk_cols = 16
top_m = 8
blocks = 2                                # number of memory+ffn blocks
ffn_mult = 2.0
Qr = 32                                   # row embedding dim (codebook factorization)
Qc = 32                                   # col embedding dim (codebook factorization)
proj_rank = 8

shared_cfg_v5 = UltraMemv5SharedCfg(
    hidden_size=H,
    n_keys=N,
    key_dim=Dk,
    tucker_rank=r,
    Rb=Rb,
    Rp=Rp,
    Qr=Qr,
    Qc=Qc,
    ks_S=ks,
    ks_T=ks,
    projector_rank=proj_rank,
)
shared_state_v5 = UltraMemv5Shared(shared_cfg_v5)

layer_cfg_v5 = UltraMemv5LayerCfg(
    hidden_size=H,
    topk_rows=topk_rows,
    topk_cols=topk_cols,
    top_m=top_m,
    softmax_tau=1.0,
)

ultra5 = UltraMemv5Classifier(
    input_dim=H,
    hidden_size=H,
    n_blocks=blocks,
    shared=shared_state_v5,
    layer_cfg=layer_cfg_v5,
    ffn_multiple=ffn_mult,
    num_classes=64,
    dropout=0.0,
)
