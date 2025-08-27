#copyright joshuah rainstar 2025 joshuah.rainstar@gmail.com
#A simplistic Recurrent MLP offers quite good, unexpected behavior
#it will beat a MOE, a GRU, etc on many toy tasks. Why?

class Cell(nn.Module):
    def __init__(self, dim_in: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden, bias=False)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='selu')
        self.fc2 = nn.Linear(hidden, dim_in, bias=True)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='selu')
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class RecurrentMLP(nn.Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.k = 3#beyond 3 ineffectual, 2 with slight drop in accuracy
        self.hidden = dim_in*2#can also use //2 with slight drop in accuracy
        self.cells_a = nn.ModuleList([Cell(dim_in, self.hidden) for _ in range(self.k)])
    def forward(self, x):
        z = x
        for i in range(self.k):
            z = z + self.cells_a[i](z)
        return z
