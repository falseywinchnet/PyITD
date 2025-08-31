#copyright joshuah rainstar 2025 joshuah.rainstar@gmail.com
#A simplistic Recurrent MLP offers quite good, unexpected behavior
#it will beat a MOE, a GRU, etc on many toy tasks. Why?
#note: marginal gain from accumulating multiple RecurrentMLP products.
#95% of work from first RMLP.

class ZLS(nn.Module):
    """
    Zero-crossing Logistic-subtracted Softplus activation
    inferior to gelu, but it offers some benefits for some problems
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp = F.softplus(x)                    # numerically stable softplus
        sa = torch.sigmoid(0.5 * x)             # s_a(x) with learned optimal shape param
        ba = sa * (1.0 - sa)                  # s_a(x) * (1 - s_a(x))
        return sp - 2.77258872223978123766 * ba #4-log(2) yields zero crossing behavior
    
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
