import game
import numpy
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim

random.seed(42)
torch.random.manual_seed(42)

print("Generating training data.")

all_inps = []
all_outs = []

for i in range(20):
    g = game.GameState()
    while not g.terminated():
        inp = g.to_image()
        legal_actions = [i for i, x in enumerate(g.legal_actions()) if x]
        out = [int(x) for i, x in enumerate(g.legal_actions())]
        action = random.choice(legal_actions)
        g = g.next_state(action)
        all_inps.append(inp[0])
        all_outs.append(out)

all_inps = torch.Tensor(all_inps)
all_outs = torch.Tensor(all_outs)

print("Training data ready. Shape:")
print(all_inps.shape)
print(all_outs.shape)

#######################################################


inp_dim = 7 * 8 * 8
hidden_dim_1 = 5000
hidden_dim_2 = 5000
hidden_dim_3 = 5000
out_dim = 64 * 64
batch_dim = 100


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.hidden_layer_1 = nn.Linear(inp_dim, hidden_dim_1)
        self.hidden_layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.hidden_layer_3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.output_layer = nn.Linear(hidden_dim_3, out_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape((batch_size, -1))
        x = F.relu(self.hidden_layer_1(x))
        x = x + F.relu(self.hidden_layer_2(x))
        x = x + F.relu(self.hidden_layer_3(x))
        x = self.output_layer(x)
        return x


print("Running neural net training!")
net = PolicyNet()
dataset = data.TensorDataset(all_inps, all_outs)
dataloader = data.DataLoader(dataset, batch_size=batch_dim, shuffle=True)

ce_loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.003)

for i in range(50):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        inps, outs = batch
        predict_outs = net(inps)
        loss = ce_loss(predict_outs, outs)
        print(f"Batch {i}: ", loss)
        loss.backward()
        optimizer.step()
