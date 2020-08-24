from torch import nn, optim


class Net(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)


    def forward(self, x):
        x_ = self.linear1(x)
        return self.linear2(x_)
