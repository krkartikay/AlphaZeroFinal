# Keras model for Alpha Zero Neural network

# Imported and used by:
#  - Selfplay.py [prediction]
#  - Evaluate.py [prediction]
#  - Train.py [training]

# Also allows saving and loading data from .h5 file.

import config
import torch
import torch.nn as nn
import torch.optim as optim

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device #
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*8*8, 64*64)
        self.fc2 = nn.Linear(64*64, 64*64)
        self.prob_logits = nn.Linear(64*64, config.num_actions)
        torch.nn.init.uniform_(self.prob_logits.weight, -0.01, 0.01)
        torch.nn.init.uniform_(self.prob_logits.bias, -0.01, 0.01)
        self.prob_head = nn.LogSoftmax(dim=1)
        self.value_head = nn.Linear(64*64, 1)
        torch.nn.init.uniform_(self.value_head.weight, -0.01, 0.01)
        torch.nn.init.uniform_(self.value_head.bias, -0.01, 0.01)
        self.value_activation = nn.Tanh()
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.loss1 = nn.KLDivLoss(reduction='batchmean')
        self.loss2 = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        log_prob = self.prob_head(self.prob_logits(x))
        value = self.value_activation(self.value_head(x))
        return log_prob, value

    def predict(self, gamestate):
        image_tensor = torch.Tensor(gamestate.to_image()).to(self.device)
        log_probs, value = self.forward(image_tensor)
        probs = torch.exp(log_probs)
        return probs, value

    def train(self, data, epochs=100, verbose=False):
        xs, probs, values = data
        xs = torch.Tensor(xs).to(self.device)
        probs = torch.Tensor(probs).to(self.device)
        values = torch.Tensor(values).to(self.device)
        loss_history = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred_log_probs, pred_values = self.forward(xs)
            pred_values = pred_values.view((-1,))
            loss1 = self.loss1(pred_log_probs, probs)
            loss2 = self.loss2(pred_values, values)
            loss = loss1 + loss2
            print(f"(#{epoch})\ttotal loss: {loss.item():.4f}, prob loss: {loss1.item():.4f}, value loss: {loss2.item():.4f}")
            loss_history.append((loss.item(), loss1.item(), loss2.item()))
            loss.backward()
            self.optimizer.step()
        return loss_history

    def load(self, filename="latest_weights.pth"):
        self.load_state_dict(torch.load(filename))
        self.to(self.device)

    def store(self, filename="latest_weights.pth"):
        torch.save(self.state_dict(), filename)
