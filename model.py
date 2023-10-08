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
        self.fc1 = nn.Linear(6*8*8, 50)
        self.fc2 = nn.Linear(50, 20)
        self.prob_logits = nn.Linear(20, config.num_actions)
        self.prob_head = nn.Softmax(dim=1)
        self.value_head = nn.Linear(20, 1)
        self.value_activation = nn.Tanh()
        self.optimizer = optim.SGD(self.parameters(), lr=config.learning_rate)
        self.loss1 = nn.KLDivLoss()
        self.loss2 = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        prob = self.prob_head(self.prob_logits(x))
        value = self.value_activation(self.value_head(x))
        return prob, value

    def predict(self, gamestate):
        image_tensor = torch.Tensor(gamestate.to_image()).to(self.device)
        return self.forward(image_tensor)

    def train_model(self, data, epochs=100, verbose=False):
        xs, probs, values = data
        xs, probs, values = xs.to(self.device), probs.to(self.device), values.to(self.device)
        loss_history = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred_probs, pred_values = self.forward(xs)
            loss = self.loss1(pred_probs, probs) + self.loss2(pred_values, values)
            print(loss)
            loss_history.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return loss_history

    def load(self, filename="latest_weights.pth"):
        self.load_state_dict(torch.load(filename))

    def store(self, filename="latest_weights.pth"):
        torch.save(self.state_dict(), filename)
