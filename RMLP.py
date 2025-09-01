#copyright joshuah rainstar 2025 joshuah.rainstar@gmail.com
#A simplistic Recurrent MLP offers quite good, unexpected behavior
#it will beat a MOE, a GRU, etc on many toy tasks. Why?
#note: marginal gain from accumulating multiple RecurrentMLP products.
#95% of work from first RMLP.
#aug 32 2025: tweaks provide incremental gains across diverse problem set.
#a stiefel projector conditions for a considerable improvement.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleStiefelProjector(nn.Module):
    def __init__(self, dim: int, t: float = 1.0, seed_angle: float = math.pi /8):
        """
        Smooth full-space rotation matrix via Lie algebra exponential map.
        A = exp(t·G), where G ∈ so(D) is skew-symmetric and full-rank.
        """
        super().__init__()
        self.dim = dim
        G = self._make_skew_symmetric_generator(dim,  math.pi /dim)#experimental change to the seed
        A = torch.matrix_exp(t * G)
        self.register_buffer("A", A)         # A ∈ SO(D)
        self.register_buffer("A_inv", A.T)   # A.T = A⁻¹

    def _make_skew_symmetric_generator(self, D, angle_scale):
        G = torch.randn(D, D)
        G = G - G.T                          # skew-symmetric: Gᵀ = -G
        norm = torch.norm(G, p='fro')
        return G * (angle_scale / norm)     # scale to control rotation strength

    def forward(self, x):
        """
        Project: x ∈ (B, D) or (B, ..., D) → x @ Aᵀ
        """
        return F.linear(x, self.A)          # A @ xᵀ

    def inverse(self, y):
        """
        Inverse: y @ A⁻¹ = y @ Aᵀ
        """
        return F.linear(y, self.A_inv)

# ---------- Block 3: Deeper MLP with 3 GELU layers, bias=True ----------
class Cell(nn.Module):
    def __init__(self, dim_in: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden, bias=False) #dont change, false intentional
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(hidden, dim_in, bias=True)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.ln = nn.LayerNorm(dim_in)

        self.act = nn.GELU()
    def forward(self, x):
        return (self.fc2(self.act(self.fc1(x))))

class RecurrentMLP(nn.Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.k = 2 #can set to 3, but marginal gains
        self.hidden = dim_in*2 #if overfitting reduce to dim_in or even dim_in//2
        self.atlas= SingleStiefelProjector(dim_in)
        self.cells_a = nn.ModuleList([Cell(dim_in, self.hidden) for _ in range(self.k)])
    def forward(self, x):
        z =  x
        for j in range(self.k):
            for i in range(self.k):
                z = z +  self.atlas.inverse(self.cells_a[j](self.atlas(z)))
        return z
