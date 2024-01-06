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
import time

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device

        self.conv0 = nn.Conv2d(7, 64, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 64*64)
        self.fc2 = nn.Linear(64 * 64, 64*64)

        self.prob_logits = nn.Linear(64*64, config.num_actions)
        self.prob_head = nn.LogSoftmax(dim=1)

        self.value_head = nn.Linear(64*64, 1)
        self.value_activation = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.loss1 = nn.KLDivLoss(reduction='batchmean')
        self.loss2 = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv0(x))

        # block 1
        x = x + nn.functional.leaky_relu(self.conv1(x))
        x = x + nn.functional.leaky_relu(self.conv2(x))
        x = x + nn.functional.leaky_relu(self.conv3(x))
        x = x + nn.functional.leaky_relu(self.conv4(x))
        x = x + nn.functional.leaky_relu(self.conv5(x))
        x = x + nn.functional.leaky_relu(self.conv6(x))
        x = x + nn.functional.leaky_relu(self.conv7(x))
        x = x + nn.functional.leaky_relu(self.conv8(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))

        log_prob = self.prob_head(self.prob_logits(x))
        value = self.value_activation(self.value_head(x))
        return log_prob, value

    def predict(self, gamestate):
        with torch.no_grad():
            image_tensor = torch.Tensor(gamestate.to_image()).to(self.device)
            log_probs, value = self.forward(image_tensor)
            probs = torch.exp(log_probs)
        return probs, value

    def train(self, data, epochs=100, verbose=False):
        xs, probs, values = data
        # xs = torch.Tensor(xs).to(self.device)
        # probs = torch.Tensor(probs).to(self.device)
        # values = torch.Tensor(values).to(self.device)
        loss_history = []
        # Create a DataLoader for batching and shuffling
        dataset = torch.utils.data.TensorDataset(xs, probs, values)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, pin_memory=True)

        for epoch in range(epochs):
            epoch_start_time = time.time()
            for i, (batch_xs, batch_probs, batch_values) in enumerate(dataloader):
                batch_start_time = time.time()
                batch_xs = batch_xs.to(self.device)
                batch_probs = batch_probs.to(self.device)
                batch_values = batch_values.to(self.device)
                # actual training steps
                self.optimizer.zero_grad()
                pred_log_probs, pred_values = self.forward(batch_xs)
                pred_values = pred_values.view((-1,))
                loss1 = self.loss1(pred_log_probs, batch_probs)
                loss2 = self.loss2(pred_values, batch_values)
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.step()
                total_batch_time = time.time() - batch_start_time
                batch_time_ms = total_batch_time * 1000
                if i % 10 == 0:
                    with torch.no_grad():
                        l = (loss.item(), loss1.item(), loss2.item())
                        print(f"(#{epoch+1:4}|{i+1:4})\ttotal loss: {loss.item():.4f}, prob loss: {loss1.item():.4f}, value loss: {loss2.item():.4f}, time/batch: {batch_time_ms:2.1f}ms")
                        loss_history.append(l)
            # note the last loss after an epoch, it causes sync issues during a batch
            # todo: we can calculate avg or total loss here somehow
            total_epoch_time = time.time() - epoch_start_time
            print(f"{total_epoch_time=}")
        return loss_history

    def load(self, filename="latest_weights.pth"):
        try:
            self.load_state_dict(torch.load(filename))
        except FileNotFoundError:
            print("No existing model!")
        self.to(self.device)

    def store(self, filename="latest_weights.pth"):
        torch.save(self.state_dict(), filename)
