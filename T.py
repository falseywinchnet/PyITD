#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution
#it came to us in a whisper on the wind
#the parseval theorem must be applied to attention
import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
# ----------------------------
# Layers
# ----------------------------

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b =self.bias if self.use_bias else None
        return F.layer_norm(x, self.weight.shape, self.weight, b, 1e-5)

# --- 1. Variance Scaled Softmax (The Statistical Repair) ---
def variance_scaled_softmax(scores, dim: int = -1, eps: float = 1e-6):
    # scores may contain -inf from masking
    finite = torch.isfinite(scores)
    m = finite.to(scores.dtype)                     # 1 where valid, 0 where masked
    n = m.sum(dim=dim, keepdim=True).clamp_min(1)   # count of valid entries per row

    # mean/var over valid entries only (population var)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))
    mean = (safe_scores * m).sum(dim=dim, keepdim=True) / n
    var  = ((safe_scores - mean)**2 * m).sum(dim=dim, keepdim=True) / n
    std  = var.clamp_min(eps).sqrt()

    # Scale to unit variance (Restoring Isometry locally)
    scaled = (safe_scores - mean) / std
    scaled = torch.where(finite, scaled, float('-inf'))  # restore mask
    
    out = torch.softmax(scaled, dim=dim)
    out = torch.where(n == 0, torch.zeros_like(out), out)  # fully-masked rows -> zeros
    return out

# --- 2. Functional Norm ---
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

# --- 3. L2 Normalization (Project to Sphere) ---
def l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# --- 4. Parseval Rotary Embedding (Single Head Adapted) ---
class ParsevalRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta_base: float = 10000.0):
        """
        dim: embedding dimension (must be even).
        """
        super().__init__()
        assert dim % 2 == 0, "dim must be even for pairing"
        self.dim = dim
        self.max_seq_len = max_seq_len

        # compute frequency for each pair
        half = dim // 2
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, half, 1, dtype=torch.float32) / half))

        # position indices
        pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (max_seq_len, 1)
        # angles (max_seq_len x half)
        angles = pos * inv_freq.unsqueeze(0)
        
        # register buffers: (1, max_seq_len, half) for broadcasting against (B, T, half)
        self.register_buffer("cos", angles.cos().unsqueeze(0)) 
        self.register_buffer("sin", angles.sin().unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_pos: torch.Tensor):
        """
        x: shape (B, T, D)
        seq_pos: tensor of positions indices shape (T,) or (B, T)
        """
        B, T, D = x.shape
        half = D // 2
        
        # Select cos/sin for current positions
        # We assume seq_pos is (T,) usually, broadcast to B if needed
        # self.cos shape: (1, max_len, half)
        # We index dim 1 with seq_pos.
        # Result shape: (1, T, half) or (B, T, half) depending on seq_pos
        
        cos_t = self.cos[:, seq_pos, :] 
        sin_t = self.sin[:, seq_pos, :]

        x1 = x[..., :half]
        x2 = x[..., half:]

        # Rotation: [x1'; x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
        x1_rot = x1 * cos_t - x2 * sin_t
        x2_rot = x1 * sin_t + x2 * cos_t

        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
        return x_rot

# --- 5. Haar Wavelet Basis Construction ---
def build_haar_wavelet_basis(T, levels, device=None, dtype=torch.float32):
    W_list = []
    for j in range(levels):
        block_count = 2**j
        block_size = T // block_count
        if block_size == 0: continue 
        half = block_size // 2
        for k in range(block_count):
            vec = torch.zeros(T, dtype=dtype, device=device)
            start = k * block_size
            mid   = start + half
            end   = start + block_size
            if half > 0:
                vec[start:mid] =  1.0 / math.sqrt(half)
                vec[mid:end]  = -1.0 / math.sqrt(half)
            W_list.append(vec)
    if len(W_list) == 0:
        # Fallback identity if T is too small for levels
        return torch.eye(T, device=device, dtype=dtype)
        
    W = torch.stack(W_list, dim=1)  # shape (T, Bcoef)
    return W

# --- 6. Single-Head Wavelet Attention (Refactored) ---
class SingleHeadWaveletAttention(nn.Module):
    def __init__(self, config, wavelet_levels=3, near_window=64):
        super().__init__()

        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.near_window = near_window
        self.wavelet_levels = wavelet_levels

        # W_Q is a learned projection.
        # We enforce Parseval by computing W_K dynamically as the metric adjoint.
        self.W_Q = nn.Parameter(torch.empty(self.n_embd, self.n_embd))
        nn.init.xavier_uniform_(self.W_Q)

        # W_V is the value projection (standard linear)
        self.W_V = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # W_O is the output projection
        self.W_O = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Precompute Haar Basis
        W_haar_full = build_haar_wavelet_basis(self.block_size,
                                               self.wavelet_levels,
                                               device='cpu')
        self.register_buffer("W_haar_full", W_haar_full)

        # Causal Mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.block_size, self.block_size))
                 .view(1, self.block_size, self.block_size) 
        )
        
        # Rotary Embedding operating on the full dimension
        self.pos_encoder = ParsevalRotaryEmbedding(dim=self.n_embd, max_seq_len=self.block_size)

    def compute_dual_WK(self):
        """
        Enforces the Operator Identity: W_Q * W_K^H = I (approximately).
        This ensures the transform to Q/K space is a frame that preserves geometry.
        """
        WQ = self.W_Q                      # (C, C)
        WQ_star = WQ.conj().T              # (C, C)
        Qmat, Rmat = torch.linalg.qr(WQ_star)
        R_inv = torch.inverse(Rmat)
        WK = R_inv @ Qmat.conj().T         # (C, C)
        return WK

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()

        # 1. Derive W_K to satisfy Parseval property
        W_K = self.compute_dual_WK()       # (C, C)

        # 2. Project and Rotary Encode
        # Note: Transposing weights for linear projection: x @ W.T
        q = x @ self.W_Q.T                 # (B, T, C)
        k = x @ W_K.T                      # (B, T, C)
        v = self.W_V(x)                    # (B, T, C)

        idx = torch.arange(T, device=x.device)
        q = self.pos_encoder(q, idx)
        k = self.pos_encoder(k, idx)

        # 3. Normalize to Sphere (Directional attention)
        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

        # 4. Near-field Mask Construction
        # (T, T) boolean mask
        near_mask_bool = (idx.view(1, -1) - idx.view(-1, 1)).abs() <= self.near_window
        
        # 5. Compute Near-field Attention (Exact)
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        # We scale by sqrt(C) purely for numerical init, though VarianceScaling will override this dynamic.
        att_near = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C))
        # Mask out far-field entries with -inf
        att_near = att_near.masked_fill(~near_mask_bool.view(1, T, T), float('-inf'))

        # 6. Compute Far-field Attention (Wavelet Compressed)
        # Slice basis for current sequence length T
        W_h_full = self.W_haar_full.to(x.device)
        W_h = W_h_full[:T, :]              # (T, Bcoef)
        
        # Project Q, K into Haar coefficients: (B, T, C) -> (B, Bcoef, C)
        # (T, Bcoef).T @ (B, T, C) -> (Bcoef, T) @ (B, T, C) 
        # We need to handle batch dim.
        # q.permute(0, 2, 1) -> (B, C, T). 
        # (B, C, T) @ W_h -> (B, C, Bcoef). Transpose back -> (B, Bcoef, C)
        q_far_proj = (q.permute(0, 2, 1) @ W_h).permute(0, 2, 1)
        k_far_proj = (k.permute(0, 2, 1) @ W_h).permute(0, 2, 1)

        # Compute Attention in Compressed Domain
        # (B, Bcoef, C) @ (B, C, Bcoef) -> (B, Bcoef, Bcoef)
        att_far_comp = (q_far_proj @ k_far_proj.transpose(-2, -1)) * (1.0 / math.sqrt(C))

        # Expand back to spatial domain
        # W_h @ (B, Bcoef, Bcoef) @ W_h.T -> (B, T, T)
        # We can use linear combination:
        # (B, Bcoef, Bcoef) -> left mult by W_h -> (B, T, Bcoef)
        # (B, T, Bcoef) -> right mult by W_h.T -> (B, T, T)
        att_far_exp = W_h @ att_far_comp @ W_h.T

        # 7. Combine Near and Far
        # Where near_mask is True, use exact near attention. Else use compressed far attention.
        att = torch.where(near_mask_bool.view(1, T, T), att_near, att_far_exp)

        # 8. Apply Causal Mask
        # self.mask is (1, Block, Block). Slice to (1, T, T).
        causal_mask = self.mask[:, :T, :T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        # 9. Apply Variance Scaled Softmax
        # This is the critical integration point for "Unitary Extended" logic.
        att = variance_scaled_softmax(att, dim=-1)

        # 10. Output Mixing
        # (B, T, T) @ (B, T, C) -> (B, T, C)
        y = att @ v
        
        # Final projection
        return self.W_O(y)
        
class UnitaryAncillaAttention(SingleHeadWaveletAttention):
    def __init__(self, config, ancilla_dim=16):
        super().__init__(config)
        self.ancilla_dim = ancilla_dim
        
        # The Ancilla is a learned orthogonal component.
        # We parameterize it as a semi-orthogonal matrix to ensure it adds 
        # pure "potential energy" without distorting the semantic direction.
        self.ancilla_param = nn.Parameter(torch.randn(1, ancilla_dim, self.n_embd))
        # Initialize to approximate orthogonality
        nn.init.orthogonal_(self.ancilla_param)

        # Augment the output projection to handle the concatenated dimension if needed,
        # but typically we project back down. 
        # For strict Parseval, we might keep dimensions, but here we fuse.

    def forward(self, x):
        B, T, C = x.size()
        
        # 1. Dual Frame W_K (Base Isometry)
        W_K = self.compute_dual_WK()
        
        # 2. Project Q, K, V
        q = x @ self.W_Q.T
        k = x @ W_K.T
        v = self.W_V(x)
        
        # 3. Orthogonal Ancilla Concatenation
        # We expand the Key/Value space with the Ancilla.
        # This effectively enlarges the Hilbert space from T to T + ancilla_dim.
        # The ancilla provides a "sink" and "source" for unitary rotation 
        # when the causal mask blocks the standard path.
        
        # Ancilla shape: (B, ancilla_dim, C)
        ancilla = self.ancilla_param.expand(B, -1, -1)
        
        # Concatenate along the Sequence dimension (Time)
        # New 'Effective Time': T_ext = T + ancilla_dim
        k_ext = torch.cat([ancilla, k], dim=1) 
        v_ext = torch.cat([ancilla, v], dim=1)
        
        # 4. Rotary Embeddings (Applied only to the semantic part 'k', or handled carefully)
        # We apply RoPE to 'q' and the semantic part of 'k'. 
        # The ancilla remains invariant or gets a fixed position code.
        idx = torch.arange(T, device=x.device)
        q = self.pos_encoder(q, idx)
        
        # Apply RoPE only to the sequence part of k_ext
        k_semantic = self.pos_encoder(k, idx)
        k_ext = torch.cat([ancilla, k_semantic], dim=1)
        
        # Normalize
        q = l2_normalize(q)
        k_ext = l2_normalize(k_ext)

        # 5. Attention with Ancilla
        # (B, T, C) @ (B, C, T+A) -> (B, T, T+A)
        scores = (q @ k_ext.transpose(-2, -1)) * (1.0 / math.sqrt(C))
        
        # 6. Extended Masking
        # The standard causal mask applies to the TxT block.
        # The Ancilla block (TxA) is ALWAYS visible (it's effectively "past" or "global" context).
        # This is crucial: The ancilla acts as the "Orthogonal Complement" storage.
        
        # Construct Mask: [Ones(T, A) | Causal(T, T)]
        # Ancilla is fully visible to all T.
        causal_mask = self.mask[:, :T, :T] # (1, T, T)
        ancilla_mask = torch.ones(1, T, self.ancilla_dim, device=x.device)
        
        # Concatenate masks
        full_mask = torch.cat([ancilla_mask, causal_mask], dim=-1)
        
        # Apply Mask
        scores = scores.masked_fill(full_mask == 0, float('-inf'))
        
        # 7. Variance Scaled Softmax
        # Now operating on the extended T+A dimension. 
        # Because 'A' is always visible, we never have zero-energy rows.
        # The energy lost from the masked future is balanced by the energy available in A.
        attn_weights = variance_scaled_softmax(scores, dim=-1)
        
        # 8. Weighted Sum
        # (B, T, T+A) @ (B, T+A, C) -> (B, T, C)
        y = attn_weights @ v_ext
        
        return self.W_O(y)


        
# ----------------------------
# Transformer Block
# ----------------------------

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear( config.n_embd,4* config.n_embd, bias=config.bias)
        self.scale = math.pi / math.sqrt(3.0)

        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class AnchorModule(nn.Module):
    """
    Learned anchor vectors + outward-normal projection.
    """
    def __init__(self, dim, n_anchor=4):
        super().__init__()
        # learned anchor points
        self.anchors = nn.Parameter(torch.randn(n_anchor, dim) / (dim ** 0.5))

    def forward(self, x):
        """
        x : (B,T,C)
        returns:
            x_out : outward-normal adjusted representation
        """

        # project x onto anchor space
        # similarity weights → soft assignment
        w = F.softmax(x @ self.anchors.t(), dim=-1)      # (B,T,n_anchor)

        # reconstruction from anchors
        recon = w @ self.anchors                        # (B,T,C)

        # residual away from manifold
        resid = x - recon                               # tangent component

        # outward-normal direction (normalized)
        norm = F.normalize(resid, dim=-1)

        # push x slightly outward from its anchor manifold
        x_out = x + resid + 0.1 * norm
        return x_out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        # new anchor before attention
        self.anchor_pre = AnchorModule(config.n_embd,32) #think from outside in :)

        self.attn = UnitaryAncillaAttention(config)

        # anchor after attention accumulation
        self.anchor_post = AnchorModule(config.n_embd,32)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # === pre-attention anchoring ===
        x_anch = self.anchor_pre(self.ln_1(x))

        # attention consumes outward-shifted x
        att = self.attn(x_anch)

        # residual update
        x = x + att

        # === re-anchor after attention ===
        x = self.anchor_post(x)#todo- can use anchor_pre here too? maybe?

        # standard MLP block
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.zeros_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, T = idx.size()

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = tok_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            
            # --- Logit Softcapping ---
            # Mathematical rationale: ln(1/epsilon) for float32 is approx 16.
            # A cap of 30 allows the model to reach "numerical certainty" without
            # exploring the unstable gradients of exp(x) where x > 50.
            # It bounds the geometric projection of the final layer.
            softcap_val = 30.0
            logits = softcap_val * torch.tanh(logits / softcap_val)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference optimization
            logits = self.lm_head(x[:, [-1], :])
            
            # Apply softcap during inference too to maintain distribution shape
            softcap_val = 30.0
            logits = softcap_val * torch.tanh(logits / softcap_val)
            
            loss = None

        return logits, loss
